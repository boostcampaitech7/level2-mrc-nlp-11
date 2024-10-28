import json, os, pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import konlpy.tag as morphs_analyzer
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluate import load
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig

from utils.common import init_obj
from utils import rank_bm25
import module.loss as module_loss
import module.metric as module_metric
import module.encoder as module_encoder
from module.data import (
    BiEncoderRetrievalPreprocDataModule,
)


class CombineBm25Retrieval:
    def __init__(self, config):
        self.config = config
        assert self.config.morphs.analyzer_name == "Kkma"
        self.tag_set = self.get_tag_set()
        self.analyzer = getattr(morphs_analyzer, self.config.morphs.analyzer_name)()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.subword.tokenizer_name
        )
        self.morphs_bm25 = None
        self.subword_bm25 = None
        (
            self.titles,
            self.contexts,
            self.tokenized_contexts,
            self.contexts_key_idx_pair,
        ) = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.morphs.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {
            v["text"]: (k, v[self.config.morphs.tokenized_column])
            for k, v in data.items()
        }
        titles, contexts, tokenized_contexts, contexts_key_idx_pair = [], [], [], {}
        for idx, (text, (k, tokenized_text)) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            contexts.append(text)
            tokenized_contexts.append(tokenized_text)
            contexts_key_idx_pair[k] = idx
        return titles, contexts, tokenized_contexts, contexts_key_idx_pair

    def tokenize(self, context):
        tokenized_context = []
        for token, pos in self.analyzer.pos(context):
            if pos in self.tag_set:
                continue
            tokenized_context.append(token)

        return tokenized_context, self.tokenizer.tokenize(context)

    def get_tag_set(self):
        if self.config.morphs.analyzer_name == "Kkma":
            josa_tag_list = (
                ["JKS", "JKC", "JKG"] + ["JKO", "JKM", "JKI"] + ["JKQ", "JC", "JX"]
            )

            suffix_list = (
                ["EPH", "EPT", "EPP", "EFN"]
                + ["EFQ", "EFO", "EFA", "EFI"]
                + ["EFR", "ECE", "ECS", "ECD"]
                + ["ETN", "ETD", "XSN", "XSV", "XSA"]
            )
            etc_list = ["IC", "SF", "SE", "SS", "SP", "SO"]
            return set(josa_tag_list + suffix_list + etc_list)
        return None

    def fit(self):
        self.morphs_bm25 = getattr(rank_bm25, self.config.morphs.model)(
            self.tokenized_contexts
        )
        del self.tokenized_contexts
        tokenized_contexts = []

        for context in self.contexts:
            tokenized_contexts.append(self.tokenizer.tokenize(context))
        self.subword_bm25 = getattr(rank_bm25, self.config.subword.model)(
            tokenized_contexts
        )

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        bm25_model_path = (
            bm25_save_dir
            + f"combine-bm25_model={self.config.morphs.model}_analyzer={self.config.morphs.analyzer_name}_tokenizer={self.config.subword.tokenizer_name}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(bm25_save_dir):
            os.makedirs(bm25_save_dir)
        with open(bm25_model_path, "wb") as file:
            self.analyzer = None
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if not self.analyzer:
            self.analyzer = getattr(morphs_analyzer, self.config.morphs.analyzer_name)()

        if isinstance(query_list, str):
            query_list = [query_list]

        tokenized_query_list = []
        for query in query_list:
            morphs_token, subword_token = self.tokenize(query)
            tokenized_query_list.append((morphs_token, subword_token))

        titles, docs_score, docs_idx, docs, query_score_list = [], [], [], [], []
        for morphs_token, subword_token in tokenized_query_list:
            morphs_doc_score, morphs_query_score = self.morphs_bm25.get_scores(
                morphs_token
            )
            subword_doc_score, subword_query_score = self.subword_bm25.get_scores(
                subword_token
            )
            doc_score = 0.5 * (morphs_doc_score / max(morphs_doc_score)) + 0.5 * (
                subword_doc_score / max(subword_doc_score)
            )

            sorted_idx = np.argsort(doc_score)[::-1]
            docs_score.append(doc_score[sorted_idx][:k])
            docs_idx.append(sorted_idx[:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            titles.append([self.titles[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append({**morphs_query_score, **subword_query_score})

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        return docs_score, docs_idx, docs, titles, query_score_list


class MorphsBm25Retrieval:
    def __init__(self, config):
        self.config = config
        assert self.config.analyzer_name == "Kkma"
        self.analyzer = getattr(morphs_analyzer, self.config.analyzer_name)()
        self.tag_set = self.get_tag_set()
        self.bm25 = None
        (
            self.titles,
            self.contexts,
            self.tokenized_contexts,
            self.contexts_key_idx_pair,
        ) = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {
            v["text"]: (k, v[self.config.tokenized_column]) for k, v in data.items()
        }
        titles, contexts, tokenized_contexts, contexts_key_idx_pair = [], [], [], {}
        for idx, (text, (k, tokenized_text)) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            contexts.append(text)
            tokenized_contexts.append(tokenized_text)
            contexts_key_idx_pair[k] = idx
        return titles, contexts, tokenized_contexts, contexts_key_idx_pair

    def tokenize(self, context):
        tokenized_context = []
        for token, pos in self.analyzer.pos(context):
            if pos in self.tag_set:
                continue
            tokenized_context.append(token)
        return tokenized_context

    def get_tag_set(self):
        if self.config.analyzer_name == "Kkma":
            josa_tag_list = (
                ["JKS", "JKC", "JKG"] + ["JKO", "JKM", "JKI"] + ["JKQ", "JC", "JX"]
            )
            suffix_list = (
                ["EPH", "EPT", "EPP", "EFN"]
                + ["EFQ", "EFO", "EFA", "EFI"]
                + ["EFR", "ECE", "ECS", "ECD"]
                + ["ETN", "ETD", "XSN", "XSV", "XSA"]
            )
            etc_list = ["IC", "SF", "SE", "SS", "SP", "SO"]
            return set(josa_tag_list + suffix_list + etc_list)
        return None

    def fit(self):
        if self.config.add_title:
            tokenized_contexts = []
            for title, tokenized_context in zip(self.titles, self.tokenized_contexts):
                tokenized_contexts.append(self.tokenize(title) + tokenized_context)
        else:
            tokenized_contexts = self.tokenized_contexts
        self.bm25 = getattr(rank_bm25, self.config.model)(tokenized_contexts)
        del self.tokenized_contexts

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        bm25_model_path = (
            bm25_save_dir
            + f"bm25-morphs_model={self.config.model}_tokenizer={self.config.analyzer_name}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(bm25_save_dir):
            os.makedirs(bm25_save_dir)
        with open(bm25_model_path, "wb") as file:
            self.analyzer = None
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if not self.analyzer:
            self.analyzer = getattr(morphs_analyzer, self.config.analyzer_name)()

        if isinstance(query_list, str):
            query_list = [query_list]

        tokenized_query_list = []
        for query in query_list:
            tokenized_query_list.append(self.tokenize(query))

        titles, docs_score, docs_idx, docs, query_score_list = [], [], [], [], []
        for tokenized_query in tokenized_query_list:
            doc_score, query_score = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(doc_score)[::-1]

            docs_score.append(doc_score[sorted_idx][:k])
            docs_idx.append(sorted_idx[:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            titles.append([self.titles[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append(query_score)

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        return docs_score, docs_idx, docs, titles, query_score_list


class SubwordBm25Retrieval:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.bm25 = None
        self.titles, self.contexts, self.contexts_key_idx_pair = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {v["text"]: k for k, v in data.items()}
        titles, contexts, contexts_key_idx_pair = [], [], {}
        for idx, (text, k) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            contexts.append(text)
            contexts_key_idx_pair[k] = idx
        return titles, contexts, contexts_key_idx_pair

    def fit(self):
        tokenized_contexts = []
        if self.config.add_title:
            for title, context in zip(self.titles, self.contexts):
                tokenized_contexts.append(
                    self.tokenizer.tokenize(title + " " + context)
                )
        else:
            for context in self.contexts:
                tokenized_contexts.append(self.tokenizer.tokenize(context))
        self.bm25 = getattr(rank_bm25, self.config.model)(tokenized_contexts)

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        bm25_model_path = (
            bm25_save_dir
            + f"bm25-subword_model={self.config.model}_tokenizer={self.config.tokenizer_name}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(bm25_save_dir):
            os.makedirs(bm25_save_dir)
        with open(bm25_model_path, "wb") as file:
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if isinstance(query_list, str):
            query_list = [query_list]

        tokenized_query_list = []
        for query in query_list:
            tokenized_query_list.append(self.tokenizer.tokenize(query))

        titles, docs_score, docs_idx, docs, query_score_list = [], [], [], [], []
        for tokenized_query in tokenized_query_list:
            doc_score, query_score = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(doc_score)[::-1]

            docs_score.append(doc_score[sorted_idx][:k])
            docs_idx.append(sorted_idx[:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            titles.append([self.titles[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append(query_score)

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        return docs_score, docs_idx, docs, titles, query_score_list


class TfIdfRetrieval:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize,
            ngram_range=tuple(self.config.ngram),
            max_features=50000,
        )
        self.sparse_embedding_matrix = None
        self.titles, self.contexts, self.contexts_key_idx_pair = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {v["text"]: k for k, v in data.items()}
        titles, contexts, contexts_key_idx_pair = [], [], {}
        for idx, (text, k) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            contexts.append(text)
            contexts_key_idx_pair[k] = idx
        return titles, np.array(contexts), contexts_key_idx_pair

    def fit(self):
        if self.config.add_title:
            contexts = [
                title + " " + context
                for title, context in zip(self.titles, self.contexts)
            ]
        else:
            contexts = self.contexts
        self.sparse_embedding_matrix = self.vectorizer.fit_transform(contexts)

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tfidf_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        tfidf_model_path = (
            tfidf_save_dir
            + f"tf-idf_tokenizer={self.config.tokenizer_name}_ngram={self.config.ngram}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(tfidf_save_dir):
            os.makedirs(tfidf_save_dir)
        with open(tfidf_model_path, "wb") as file:
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if isinstance(query_list, str):
            query_list = [query_list]

        titles, docs_score, docs_idx, docs = [], [], [], []
        query_vector_list = self.vectorizer.transform(query_list)
        similarity = query_vector_list * self.sparse_embedding_matrix.T

        for i in range(len(query_list)):
            sorted_idx = np.argsort(similarity[i].data)[::-1]
            doc_idx = similarity[i].indices[sorted_idx][:k]
            doc_score = similarity[i].data[sorted_idx][:k]

            docs_score.append(doc_score)
            docs_idx.append(doc_idx)
            docs.append(list(self.contexts[doc_idx]))
            titles.append([self.titles[idx] for idx in doc_idx])

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        query_score_list = []
        vocab = sorted(self.vectorizer.vocabulary_.keys())
        sparse_embedding_matrix = self.sparse_embedding_matrix.toarray()

        for query_vector in query_vector_list:
            q_scores = defaultdict(lambda: np.zeros(sparse_embedding_matrix.shape[0]))
            for idx, data in zip(query_vector.indices, query_vector.data):
                q_scores[vocab[idx]] += data * sparse_embedding_matrix[:, idx]
            query_score_list.append(q_scores)

        return docs_score, docs_idx, docs, titles, query_score_list


class BiEncoderDenseRetrieval(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.preprocess_module = BiEncoderRetrievalPreprocDataModule(config)
        self.dense_emb_matrix = None
        self.titles, self.contexts, self.contexts_key_idx_pair = [], [], {}

        if self.config.model.use_lora:
            lora_config = LoraConfig(
                r=8,  # 랭크 값, 모델이 얼마나 적은 파라미터로 학습할지 결정
                lora_alpha=32,  # LoRA 알파 값
                target_modules=[
                    "query",
                    "key",
                ],  # LoRA 적용할 레이어. 예를 들어 Self-Attention의 query, key에 적용
                lora_dropout=0.1,  # Dropout 확률
                bias="none",  # 편향을 LoRA에 포함할지 여부 (none, all, lora-only 중 선택 가능)
            )

        self.save_hyperparameters()
        if self.config.model.use_single_model:
            encoder = (
                getattr(module_encoder, self.config.model.encoder)
                .from_pretrained(self.config.model.plm_name)
                .to(self.config.device)
            )
            if self.config.model.use_lora:
                encoder = get_peft_model(encoder, lora_config)
            self.q_encoder = self.p_encoder = encoder
        else:
            self.q_encoder = (
                getattr(module_encoder, self.config.model.encoder)
                .from_pretrained(self.config.model.plm_name)
                .to(self.config.device)
            )
            self.p_encoder = (
                getattr(module_encoder, self.config.model.encoder)
                .from_pretrained(self.config.model.plm_name)
                .to(self.config.device)
            )
            if self.config.model.use_lora:
                self.q_encoder = get_peft_model(self.q_encoder, lora_config)
                self.p_encoder = get_peft_model(self.p_encoder, lora_config)

        self.criterion = getattr(module_loss, self.config.loss)
        self.validation_step_outputs = {"sim_score": [], "targets": []}
        self.metric_list = {
            metric: {"method": load(metric), "wrapper": getattr(module_metric, metric)}
            for metric in self.config.metric
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyper_parameters"]["dense_emb_matrix"] = self.dense_emb_matrix
        checkpoint["hyper_parameters"]["titles"] = self.titles
        checkpoint["hyper_parameters"]["contexts"] = self.contexts
        checkpoint["hyper_parameters"][
            "contexts_key_idx_pair"
        ] = self.contexts_key_idx_pair
        (
            self.dense_emb_matrix,
            self.titles,
            self.contexts,
            self.contexts_key_idx_pair,
        ) = (None, [], [], {})

    def on_load_checkpoint(self, checkpoint):
        self.dense_emb_matrix = checkpoint["hyper_parameters"]["dense_emb_matrix"]
        self.titles = checkpoint["hyper_parameters"]["titles"]
        self.contexts = checkpoint["hyper_parameters"]["contexts"]
        self.contexts_key_idx_pair = checkpoint["hyper_parameters"][
            "contexts_key_idx_pair"
        ]

    def configure_optimizers(self):
        if self.config.model.use_single_model:
            trainable_params = [
                {
                    "params": list(
                        filter(lambda p: p.requires_grad, self.q_encoder.parameters())
                    )
                }
            ]
        else:
            trainable_params1 = list(
                filter(lambda p: p.requires_grad, self.q_encoder.parameters())
            )
            trainable_params2 = list(
                filter(lambda p: p.requires_grad, self.p_encoder.parameters())
            )
            trainable_params = [
                {"params": trainable_params1},
                {"params": trainable_params2},
            ]

        optimizer_name = self.config.optimizer.name
        del self.config.optimizer.name
        optimizer = init_obj(
            optimizer_name, self.config.optimizer, torch.optim, trainable_params
        )
        return optimizer

    def training_step(self, batch):
        targets = torch.zeros(self.config.data.batch_size).long().to(self.config.device)

        p_inputs = {
            "input_ids": batch[0]
            .view(-1, self.config.data.max_seq_length)
            .to(self.config.device),
            "attention_mask": batch[1]
            .view(-1, self.config.data.max_seq_length)
            .to(self.config.device),
            "token_type_ids": batch[2]
            .view(-1, self.config.data.max_seq_length)
            .to(self.config.device),
        }

        q_inputs = {
            "input_ids": batch[3].to(self.config.device),
            "attention_mask": batch[4].to(self.config.device),
            "token_type_ids": batch[5].to(self.config.device),
        }

        overflow_size = batch[6].to(self.config.device)  # bz x num_neg + 1

        p_outputs = self.p_encoder(**p_inputs)
        q_outputs = self.q_encoder(**q_inputs)

        p_outputs = p_outputs.view(
            self.config.data.batch_size,
            self.config.data.num_neg + 1,
            -1,
            p_outputs.size()[-1],
        )
        mean_p_outputs = torch.sum(p_outputs, dim=-2) / overflow_size.unsqueeze(-1)
        q_outputs = q_outputs.view(self.config.data.batch_size, 1, -1)

        similarity_scores = torch.bmm(
            q_outputs, mean_p_outputs.transpose(-2, -1)
        ).squeeze()
        similarity_scores = F.log_softmax(similarity_scores, dim=-1)

        loss = self.criterion(similarity_scores, targets)
        self.log("step_train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch):
        targets = torch.zeros(self.config.data.batch_size).long().to(self.config.device)

        p_inputs = {
            "input_ids": batch[0]
            .view(-1, self.config.data.max_seq_length)
            .to(self.config.device),
            "attention_mask": batch[1]
            .view(-1, self.config.data.max_seq_length)
            .to(self.config.device),
            "token_type_ids": batch[2]
            .view(-1, self.config.data.max_seq_length)
            .to(self.config.device),
        }

        q_inputs = {
            "input_ids": batch[3].to(self.config.device),
            "attention_mask": batch[4].to(self.config.device),
            "token_type_ids": batch[5].to(self.config.device),
        }

        overflow_size = batch[6].to(self.config.device)  # bz x num_neg + 1

        p_outputs = self.p_encoder(**p_inputs)
        q_outputs = self.q_encoder(**q_inputs)

        p_outputs = p_outputs.view(
            self.config.data.batch_size,
            self.config.data.num_neg + 1,
            -1,
            p_outputs.size()[-1],
        )
        mean_p_outputs = torch.sum(p_outputs, dim=-2) / overflow_size.unsqueeze(-1)
        q_outputs = q_outputs.view(self.config.data.batch_size, 1, -1)

        similarity_scores = torch.bmm(
            q_outputs, mean_p_outputs.transpose(-2, -1)
        ).squeeze()
        similarity_scores = F.log_softmax(similarity_scores, dim=-1)

        self.validation_step_outputs["sim_score"].extend(similarity_scores.cpu())
        self.validation_step_outputs["targets"].extend(targets.cpu())

    def on_validation_epoch_end(self):
        for k, v in self.validation_step_outputs.items():
            self.validation_step_outputs[k] = np.array(v).squeeze()

        # compute metric
        for name, metric in self.metric_list.items():
            metric_result = metric["wrapper"](
                self.validation_step_outputs, metric["method"]
            )
            for k, v in metric_result.items():
                self.log(k, v)
        self.validation_step_outputs = {"sim_score": [], "targets": []}

    def create_contexts_emb_vec(self):
        # 1. 문서 데이터 로드
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data.context_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {v["text"]: k for k, v in data.items()}
        for idx, (text, k) in enumerate(text_key_pair.items()):
            self.titles.append(data[k]["title"])
            self.contexts.append(text)
            self.contexts_key_idx_pair[k] = idx

        # 2. 문서 데이터 임베딩
        self.p_encoder.eval()
        self.p_encoder = self.p_encoder.to(self.config.device)
        contexts_emb = []

        if self.config.data.use_overflow_token:
            tokenized_contexts, overflow_size = (
                self.preprocess_module.process_overflow_token(self.contexts)
            )
        else:
            tokenized_contexts, overflow_size = (
                self.preprocess_module.cut_overflow_token(self.contexts)
            )
            self.config.data.overflow_limit = 1

        offset = 30 * self.config.data.overflow_limit * self.config.data.batch_size
        total_len = len(tokenized_contexts["input_ids"])

        with torch.no_grad():
            for i in tqdm(range(0, total_len // offset + 1)):
                sub_tokenized_contexts = dict()
                sub_tokenized_contexts["input_ids"] = (
                    torch.tensor(
                        tokenized_contexts["input_ids"][offset * i : offset * (i + 1)]
                    )
                    .view(-1, self.config.data.max_seq_length)
                    .to(self.config.device)
                    .long()
                )
                sub_tokenized_contexts["attention_mask"] = (
                    torch.tensor(
                        tokenized_contexts["attention_mask"][
                            offset * i : offset * (i + 1)
                        ]
                    )
                    .view(-1, self.config.data.max_seq_length)
                    .to(self.config.device)
                    .long()
                )
                sub_tokenized_contexts["token_type_ids"] = (
                    torch.tensor(
                        tokenized_contexts["token_type_ids"][
                            offset * i : offset * (i + 1)
                        ]
                    )
                    .view(-1, self.config.data.max_seq_length)
                    .to(self.config.device)
                    .long()
                )
                sub_overflow_size = torch.tensor(
                    overflow_size[
                        (offset // self.config.data.overflow_limit)
                        * i : (offset // self.config.data.overflow_limit)
                        * (i + 1)
                    ]
                ).to(self.config.device)
                context_emb = self.p_encoder(**sub_tokenized_contexts)
                mean_context_emb = torch.sum(
                    context_emb.view(
                        -1, self.config.data.overflow_limit, context_emb.size()[-1]
                    ),
                    dim=-2,
                ) / sub_overflow_size.unsqueeze(-1)
                contexts_emb.append(mean_context_emb)

            self.dense_emb_matrix = torch.cat(contexts_emb).T
            print(f"dense embedding matrix shape: {self.dense_emb_matrix.size()}")

    def search(self, query, k=1, return_sim_score=False):
        self.q_encoder.eval()
        self.q_encoder = self.q_encoder.to(self.config.device)

        tokenized_query = self.preprocess_module.tokenizer(
            [query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.config.data.max_seq_length,
        )
        with torch.no_grad():
            query_vector = self.q_encoder(**tokenized_query.to(self.config.device))

        similarity_score = torch.matmul(query_vector, self.dense_emb_matrix).squeeze()
        sorted_idx = torch.argsort(similarity_score, dim=-1, descending=True).squeeze()
        doc_scores = similarity_score[sorted_idx]

        if return_sim_score:
            return (
                doc_scores[:k].cpu(),
                sorted_idx[:k].cpu(),
                [self.contexts[idx] for idx in sorted_idx[:k]],
                [self.titles[idx] for idx in sorted_idx[:k]],
                similarity_score.cpu(),
            )

        return (
            doc_scores[:k].cpu(),
            sorted_idx[:k].cpu(),
            [self.contexts[idx] for idx in sorted_idx[:k]],
            [self.titles[idx] for idx in sorted_idx[:k]],
        )


def create_and_save_bi_encoder_emb_matrix(checkpoint_path):
    retrieval = BiEncoderDenseRetrieval.load_from_checkpoint(
        checkpoint_path, strict=False
    )
    retrieval.create_contexts_emb_vec()

    trainer = pl.Trainer()
    trainer.strategy.connect(retrieval)
    trainer.save_checkpoint(checkpoint_path.replace(".ckpt", "_emb-vec.ckpt"))


class RetrievalReranker:

    def __init__(self, sparse_retrieval, dense_retrieval):
        self.sparse_retrieval = sparse_retrieval
        self.dense_retrieval = dense_retrieval
        self.sparse_retrieval_contexts_idx_key_pair = {
            v: k for k, v in self.sparse_retrieval.contexts_key_idx_pair.items()
        }
        self.dense_retrieval_contexts_key_idx_pair = (
            self.dense_retrieval.contexts_key_idx_pair
        )

    def search(self, question, rerank_k=100, final_k=10):
        docs_score, docs_idx, docs, titles = self.sparse_retrieval.search(
            question, k=rerank_k
        )
        docs_key = [
            self.sparse_retrieval_contexts_idx_key_pair[doc_idx]
            for doc_idx in docs_idx[0]
        ]
        sparse_docs_key_score_pair = {
            doc_key: doc_score for doc_key, doc_score in zip(docs_key, docs_score[0])
        }
        key_doc_pair = {
            doc_key: (doc, title)
            for doc_key, doc, title in zip(docs_key, docs[0], titles[0])
        }
        sparse_doc_score_max = np.max(docs_score)

        dense_retrieval_docs_idx = [
            self.dense_retrieval_contexts_key_idx_pair[key] for key in docs_key
        ]
        _, _, _, _, sim_score = self.dense_retrieval.search(
            question, k=rerank_k, return_sim_score=True
        )
        dense_docs_key_score_pair = {
            doc_key: doc_score
            for doc_key, doc_score in zip(docs_key, sim_score[dense_retrieval_docs_idx])
        }
        dense_doc_score_max = np.max(sim_score[dense_retrieval_docs_idx].numpy())

        final_docs_key_score_pair = {
            doc_key: 0.5 * (sparse_docs_key_score_pair[doc_key] / sparse_doc_score_max)
            + 0.5 * (dense_docs_key_score_pair[doc_key] / dense_doc_score_max)
            for doc_key in docs_key
        }
        final_docs_key_score_pair = sorted(
            final_docs_key_score_pair.items(), key=lambda item: -item[1]
        )[:final_k]

        final_docs_score, final_docs_idx, final_docs, final_titles = [], [], [], []
        for doc_key, score in final_docs_key_score_pair:
            final_docs_score.append(score)
            final_docs_idx.append(self.dense_retrieval_contexts_key_idx_pair[doc_key])
            final_docs.append(key_doc_pair[doc_key][0])
            final_titles.append(key_doc_pair[doc_key][1])

        return final_docs_score, final_docs_idx, final_docs, final_titles
