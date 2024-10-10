import collections, json, logging, os
from typing import Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import (
    AutoModelForQuestionAnswering,
    EvalPrediction,
    RobertaForQuestionAnswering,
)
from datasets import load_metric, load_dataset, Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn.functional as F
import torchmetrics
import module.loss as module_loss
from utils.common import init_obj
import rank_bm25
import konlpy.tag as morphs_analyzer

from utils.data_template import get_dataset_list
from evaluate import load
from utils.common import init_obj
import module.metric as module_metric


def identity_func(x):
    return x


class Bm25Retrieval:

    def __init__(self, config):
        self.config = config
        try:
            self.tokenize_func = AutoTokenizer.from_pretrained(
                self.config.bm25.tokenizer
            ).tokenize
        except Exception as e:
            self.tokenize_func = getattr(
                morphs_analyzer, self.config.bm25.tokenizer
            )().morphs

        self.bm25 = None
        self.contexts = self.prepare_contexts()

    def prepare_contexts(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.bm25.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # case 1. pretrained model tokenizer
        if self.tokenize_func.__name__ == "tokenize":
            return list(dict.fromkeys([v["text"] for v in data.values()]))
        # case 2. morphs analyzer
        elif self.tokenize_func.__name__ == "morphs":
            ST = {v["text"]: v["morphs_text"] for v in data.values()}
            self.morphs_contexts = [morphs_text for morphs_text in ST.values()]
            return [text for text in ST.keys()]

    def fit(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/bm25/"
        bm25_model_path = (
            bm25_save_dir
            + f"{self.config.bm25.model}_{self.config.bm25.tokenizer}".replace("/", "-")
        )

        if os.path.isfile(bm25_model_path):
            with open(bm25_model_path, "rb") as file:
                self.bm25 = pickle.load(file)
        else:
            # case 1. pretrained model tokenizer
            if self.tokenize_func.__name__ == "tokenize":
                tokenized_contexts = []
                for context in self.contexts:
                    tokenized_contexts.append(self.tokenize_func(context))
                self.bm25 = getattr(rank_bm25, self.config.bm25.model)(
                    tokenized_contexts
                )
            # case 2. morphs analyzer
            elif self.tokenize_func.__name__ == "morphs":
                self.bm25 = getattr(rank_bm25, self.config.bm25.model)(
                    self.morphs_contexts
                )

            if not os.path.isdir(bm25_save_dir):
                os.makedirs(bm25_save_dir)
            with open(bm25_model_path, "wb") as file:
                pickle.dump(self.bm25, file)

        if hasattr(self, "morphs_contexts"):
            del self.morphs_contexts

    def search(self, query, k=1):
        if isinstance(query, str):
            query = [query]

        tokenized_query_list = []
        for q in query:
            tokenized_query_list.append(self.tokenize_func(q))

        docs_score, docs = [], []
        for tokenized_query in tokenized_query_list:
            doc_scores = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(doc_scores)[::-1]
            docs_score.append(doc_scores[sorted_idx][:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])

        return docs_score, docs


class TfIdfRetrieval:

    def __init__(self, config):
        self.config = config
        try:
            self.tokenize_func = AutoTokenizer.from_pretrained(
                self.config.tfidf.tokenizer
            ).tokenize
        except Exception as e:
            self.tokenize_func = getattr(
                morphs_analyzer, self.config.tfidf.tokenizer
            )().morphs

        self.contexts = self.prepare_contexts()
        self.vectorizer = TfidfVectorizer(
            tokenizer=identity_func,
            lowercase=False,
            ngram_range=tuple(self.config.tfidf.ngram),
            max_features=50000,
        )
        self.sparse_embedding_matrix = None

    def prepare_contexts(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.tfidf.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # case 1. pretrained model tokenizer
        if self.tokenize_func.__name__ == "tokenize":
            return np.array(list(dict.fromkeys([v["text"] for v in data.values()])))
        # case 2. morphs analyzer
        elif self.tokenize_func.__name__ == "morphs":
            ST = {v["text"]: v["morphs_text"] for v in data.values()}
            self.morphs_contexts = [morphs_text for morphs_text in ST.values()]
            return np.array([text for text in ST.keys()])

    def fit(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tfidf_save_dir = f"{parent_directory}/tfidf/"
        tfidf_model_path = (
            tfidf_save_dir
            + f"model_{self.config.tfidf.tokenizer}_ngram={self.config.tfidf.ngram}".replace(
                "/", "-"
            )
        )
        tfidf_embed_path = (
            tfidf_save_dir
            + f"embed_{self.config.tfidf.tokenizer}_ngram={self.config.tfidf.ngram}".replace(
                "/", "-"
            )
        )

        # 1. load model and matrix
        if os.path.isfile(tfidf_model_path) and os.path.isfile(tfidf_embed_path):
            with open(tfidf_model_path, "rb") as file:
                self.vectorizer = pickle.load(file)
            with open(tfidf_embed_path, "rb") as file:
                self.sparse_embedding_matrix = pickle.load(file)
        # 2. create model and matrix
        else:
            # case 1. pretrained model tokenizer
            tokenized_contexts = []
            if self.tokenize_func.__name__ == "tokenize":
                for context in self.contexts:
                    tokenized_contexts.append(self.tokenize_func(context))
            # case 2. morphs analyzer
            elif self.tokenize_func.__name__ == "morphs":
                for morphs_context in self.morphs_contexts:
                    tokenized_contexts.append(morphs_context)
            self.sparse_embedding_matrix = self.vectorizer.fit_transform(
                tokenized_contexts
            )

            # save model and matrix
            if not os.path.isdir(tfidf_save_dir):
                os.makedirs(tfidf_save_dir)
            with open(tfidf_model_path, "wb") as file:
                pickle.dump(self.vectorizer, file)
            with open(tfidf_embed_path, "wb") as file:
                pickle.dump(self.sparse_embedding_matrix, file)

        if hasattr(self, "morphs_contexts"):
            del self.morphs_contexts

    def search(self, query, k=1):
        if isinstance(query, str):
            query = [query]

        tokenized_query_list = []
        for q in query:
            tokenized_query_list.append(self.tokenize_func(q))

        doc_scores, docs = [], []
        query_vector = self.vectorizer.transform(tokenized_query_list)
        similarity = query_vector * self.sparse_embedding_matrix.T
        for i in range(len(query)):
            sorted_idx = np.argsort(similarity[i].data)[::-1]
            doc_scores.append(similarity[i].data[sorted_idx][:k])
            docs.append(list(self.contexts[similarity[i].indices[sorted_idx][:k]]))

        return doc_scores, docs


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
