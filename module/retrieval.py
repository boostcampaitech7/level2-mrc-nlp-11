import json, os, pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import concatenate_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluate import load
from transformers import AutoTokenizer
import konlpy.tag as morphs_analyzer

import module.loss as module_loss
from utils.common import init_obj
from utils import rank_bm25
from utils.data_template import get_dataset_list
from utils.common import init_obj
import module.metric as module_metric


class MorphsBm25Retrieval:
    def __init__(self, config):
        self.config = config
        assert self.config.analyzer_name == "Kkma"
        self.analyzer = getattr(morphs_analyzer, self.config.analyzer_name)()
        self.tag_set = self.get_tag_set()
        self.bm25 = None
        self.contexts, self.tokenized_contexts, self.contexts_key_idx_pair = (
            self.prepare_contexts()
        )

    def prepare_contexts(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {
            v["text"]: (k, v[self.config.tokenized_column]) for k, v in data.items()
        }
        contexts, tokenized_contexts, contexts_key_idx_pair = [], [], {}
        for idx, (text, (k, tokenized_text)) in enumerate(text_key_pair.items()):
            contexts.append(text)
            tokenized_contexts.append(tokenized_text)
            contexts_key_idx_pair[k] = idx
        return contexts, tokenized_contexts, contexts_key_idx_pair

    def tokenize(self, context):
        self.analyzer.pos(context)
        tokenized_context = []
        for token, pos in self.analyzer.pos(context):
            if pos in self.tag_set:
                continue
            tokenized_context.append(token)
        return tokenized_context

    def get_tag_set(self):
        if self.config.analyzer_name == "Kkma":
            josa_tag_list = [
                "JKS",
                "JKC",
                "JKG",
                "JKO",
                "JKM",
                "JKI",
                "JKQ",
                "JC",
                "JX",
            ]
            suffix_list = [
                "EPH",
                "EPT",
                "EPP",
                "EFN",
                "EFQ",
                "EFO",
                "EFA",
                "EFI",
                "EFR",
                "ECE",
                "ECS",
                "ECD",
                "ETN",
                "ETD",
                "XSN",
                "XSV",
                "XSA",
            ]
            etc_list = ["IC", "SF", "SE", "SS", "SP", "SO"]
            return set(josa_tag_list + suffix_list + etc_list)
        return None

    def fit(self):
        self.bm25 = getattr(rank_bm25, self.config.model)(self.tokenized_contexts)
        del self.tokenized_contexts

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoint/"
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

        docs_score, docs_idx, docs, query_score_list = [], [], [], []
        for tokenized_query in tokenized_query_list:
            doc_score, query_score = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(doc_score)[::-1]

            docs_idx.append(sorted_idx[:k])
            docs_score.append(doc_score[sorted_idx][:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append(query_score)

        if not return_query_score:
            return docs_score, docs_idx, docs

        return docs_score, docs_idx, docs, query_score_list


class SubwordBm25Retrieval:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.bm25 = None
        self.contexts, self.contexts_key_idx_pair = self.prepare_contexts()

    def prepare_contexts(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {v["text"]: k for k, v in data.items()}
        contexts, contexts_key_idx_pair = [], {}
        for idx, (text, k) in enumerate(text_key_pair.items()):
            contexts.append(text)
            contexts_key_idx_pair[k] = idx
        return contexts, contexts_key_idx_pair

    def fit(self):
        tokenized_contexts = []
        for context in self.contexts:
            tokenized_contexts.append(self.tokenizer.tokenize(context))
        self.bm25 = getattr(rank_bm25, self.config.model)(tokenized_contexts)

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoint/"
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

        docs_score, docs_idx, docs, query_score_list = [], [], [], []
        for tokenized_query in tokenized_query_list:
            doc_score, query_score = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(doc_score)[::-1]

            docs_idx.append(sorted_idx[:k])
            docs_score.append(doc_score[sorted_idx][:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append(query_score)

        if not return_query_score:
            return docs_score, docs_idx, docs

        return docs_score, docs_idx, docs, query_score_list


class TfIdfRetrieval:

    def __init__(self, config):
        self.config = config
        self.contexts, self.contexts_key_idx_pair = self.prepare_contexts()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize,
            ngram_range=tuple(self.config.ngram),
            max_features=50000,
        )
        self.sparse_embedding_matrix = None

    def prepare_contexts(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {v["text"]: k for k, v in data.items()}
        contexts, contexts_key_idx_pair = [], {}
        for idx, (text, k) in enumerate(text_key_pair.items()):
            contexts.append(text)
            contexts_key_idx_pair[k] = idx
        return np.array(contexts), contexts_key_idx_pair

    def fit(self):
        self.sparse_embedding_matrix = self.vectorizer.fit_transform(self.contexts)

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tfidf_save_dir = f"{parent_directory}/retrieval_checkpoint/"
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

        docs_score, docs_idx, docs = [], [], []
        query_vector_list = self.vectorizer.transform(query_list)
        similarity = query_vector_list * self.sparse_embedding_matrix.T

        for i in range(len(query_list)):
            sorted_idx = np.argsort(similarity[i].data)[::-1]
            doc_idx = similarity[i].indices[sorted_idx][:k]
            doc_score = similarity[i].data[sorted_idx][:k]

            docs.append(list(self.contexts[doc_idx]))
            docs_idx.append(doc_idx)
            docs_score.append(doc_score)

        if not return_query_score:
            return docs_score, docs_idx, docs

        query_score_list = []
        vocab = sorted(self.vectorizer.vocabulary_.keys())
        sparse_embedding_matrix = self.sparse_embedding_matrix.toarray()

        for query_vector in query_vector_list:
            q_scores = defaultdict(lambda: np.zeros(sparse_embedding_matrix.shape[0]))
            for idx, data in zip(query_vector.indices, query_vector.data):
                q_scores[vocab[idx]] += data * sparse_embedding_matrix[:, idx]
            query_score_list.append(q_scores)

        return docs_score, docs_idx, docs, query_score_list


class DenseRetrieval(pl.LightningModule):

    def __init__(self, config, q_encoder, p_encoder):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)
        self.q_encoder = q_encoder.to(self.config.device)
        self.p_encoder = p_encoder.to(self.config.device)
        self.dense_embedding_matrix = None
        self.criterion = getattr(module_loss, self.config.loss)
        self.validation_step_outputs = {"sim_score": [], "targets": []}
        self.metric_list = {
            metric: {"method": load(metric), "wrapper": getattr(module_metric, metric)}
            for metric in self.config.metric
        }

    def configure_optimizers(self):
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
            .view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1)
            .to(self.config.device),
            "attention_mask": batch[1]
            .view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1)
            .to(self.config.device),
            "token_type_ids": batch[2]
            .view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1)
            .to(self.config.device),
        }

        q_inputs = {
            "input_ids": batch[3].to(self.config.device),
            "attention_mask": batch[4].to(self.config.device),
            "token_type_ids": batch[5].to(self.config.device),
        }

        p_outputs = self.p_encoder(**p_inputs)
        q_outputs = self.q_encoder(**q_inputs)

        p_outputs = p_outputs.view(
            self.config.data.batch_size, self.config.data.num_neg + 1, -1
        )
        q_outputs = q_outputs.view(self.config.data.batch_size, 1, -1)

        similarity_scores = torch.bmm(q_outputs, p_outputs.transpose(-2, -1)).squeeze()
        similarity_scores = F.log_softmax(similarity_scores, dim=-1)

        loss = self.criterion(similarity_scores, targets)
        self.log("step_train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch):
        targets = torch.zeros(self.config.data.batch_size).long().to(self.config.device)

        p_inputs = {
            "input_ids": batch[0]
            .view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1)
            .to(self.config.device),
            "attention_mask": batch[1]
            .view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1)
            .to(self.config.device),
            "token_type_ids": batch[2]
            .view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1)
            .to(self.config.device),
        }

        q_inputs = {
            "input_ids": batch[3].to(self.config.device),
            "attention_mask": batch[4].to(self.config.device),
            "token_type_ids": batch[5].to(self.config.device),
        }

        p_outputs = self.p_encoder(**p_inputs)
        q_outputs = self.q_encoder(**q_inputs)

        p_outputs = p_outputs.view(
            self.config.data.batch_size, self.config.data.num_neg + 1, -1
        )
        q_outputs = q_outputs.view(self.config.data.batch_size, 1, -1)

        similarity_scores = torch.bmm(q_outputs, p_outputs.transpose(-2, -1)).squeeze()
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

    def create_embedding_vector(self):
        self.p_encoder.eval()
        dataset_list = get_dataset_list(self.config.data.dataset_name)

        eval_dataset = concatenate_datasets(
            [ds["validation"] for ds in dataset_list]
        ).select(range(100))
        self.eval_corpus = list(set([example["context"] for example in eval_dataset]))

        p_embs = []
        for p in self.eval_corpus:
            p = self.tokenizer(
                p,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.config.data.max_seq_length,
            )
            p_emb = self.p_encoder(**p)
            p_embs.append(p_emb.squeeze())

        self.dense_embedding_matrix = torch.stack(p_embs).T

    def search(self, query, k=1):
        self.q_encoder.eval()

        query_token = self.tokenizer(
            [query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.config.data.max_seq_length,
        )
        query_vector = self.q_encoder(**query_token)

        similarity_score = torch.matmul(
            query_vector, self.dense_embedding_matrix
        ).squeeze()
        sorted_idx = torch.argsort(similarity_score, dim=-1, descending=True).squeeze()
        doc_scores = similarity_score[sorted_idx]

        return doc_scores[:k], self.eval_corpus[sorted_idx[:k]]
