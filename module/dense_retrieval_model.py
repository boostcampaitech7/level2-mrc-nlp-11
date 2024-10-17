import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from evaluate import load
from transformers import AutoTokenizer

import module.loss as module_loss
from utils.common import init_obj
import module.metric as module_metric
import module.dense_retrieval_encoder as module_encoder


class BiEncoderDenseRetrieval(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)
        if self.config.model.use_single_model:
            self.q_encoder = self.p_encoder = (
                getattr(module_encoder, self.config.model.encoder)
                .from_pretrained(self.config.model.plm_name)
                .to(self.config.device)
            )

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
        self.dense_emb_matrix = None
        self.contexts = None
        self.criterion = getattr(module_loss, self.config.loss)
        self.validation_step_outputs = {"sim_score": [], "targets": []}
        self.metric_list = {
            metric: {"method": load(metric), "wrapper": getattr(module_metric, metric)}
            for metric in self.config.metric
        }

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
            .view(
                self.config.data.batch_size
                * (self.config.data.num_neg + 1)
                * self.config.data.overflow_limit,
                -1,
            )
            .to(self.config.device),
            "attention_mask": batch[1]
            .view(
                self.config.data.batch_size
                * (self.config.data.num_neg + 1)
                * self.config.data.overflow_limit,
                -1,
            )
            .to(self.config.device),
            "token_type_ids": batch[2]
            .view(
                self.config.data.batch_size
                * (self.config.data.num_neg + 1)
                * self.config.data.overflow_limit,
                -1,
            )
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
            self.config.data.batch_size,
            self.config.data.num_neg + 1,
            self.config.data.overflow_limit,
            -1,
        )
        mean_p_outputs = torch.sum(p_outputs, dim=-2)
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
            .view(
                self.config.data.batch_size
                * (self.config.data.num_neg + 1)
                * self.config.data.overflow_limit,
                -1,
            )
            .to(self.config.device),
            "attention_mask": batch[1]
            .view(
                self.config.data.batch_size
                * (self.config.data.num_neg + 1)
                * self.config.data.overflow_limit,
                -1,
            )
            .to(self.config.device),
            "token_type_ids": batch[2]
            .view(
                self.config.data.batch_size
                * (self.config.data.num_neg + 1)
                * self.config.data.overflow_limit,
                -1,
            )
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
            self.config.data.batch_size,
            self.config.data.num_neg + 1,
            self.config.data.overflow_limit,
            -1,
        )
        mean_p_outputs = torch.sum(p_outputs, dim=-2)
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
        self.p_encoder.eval()
        self.contexts = None
        self.dense_emb_matrix = None
        pass

    def search(self, query, k=1):
        self.q_encoder.eval()

        tokenized_query = self.tokenizer(
            [query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.config.data.max_seq_length,
        )
        query_vector = self.q_encoder(**tokenized_query)

        similarity_score = torch.matmul(query_vector, self.dense_emb_matrix).squeeze()
        sorted_idx = torch.argsort(similarity_score, dim=-1, descending=True).squeeze()
        doc_scores = similarity_score[sorted_idx]

        return doc_scores[:k], self.contexts[sorted_idx[:k]]


class CrossEncoderDenseRetrieval(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)
        self.encoder = module_encoder.CrossEncoder(self.config.model.plm_name).to(
            self.config.device
        )
        self.dense_emb_matrix = None
        self.contexts = None
        self.criterion = getattr(module_loss, self.config.loss)
        self.validation_step_outputs = {"sim_score": [], "targets": []}
        self.metric_list = {
            metric: {"method": load(metric), "wrapper": getattr(module_metric, metric)}
            for metric in self.config.metric
        }

    def configure_optimizers(self):
        trainable_params1 = list(
            filter(lambda p: p.requires_grad, self.encoder.parameters())
        )
        trainable_params = [
            {"params": trainable_params1},
        ]

        optimizer_name = self.config.optimizer.name
        del self.config.optimizer.name
        optimizer = init_obj(
            optimizer_name, self.config.optimizer, torch.optim, trainable_params
        )
        return optimizer

    def training_step(self, batch):
        targets = torch.zeros(self.config.data.batch_size).long().to(self.config.device)
        input_args = ["input_ids", "attention_mask", "token_type_ids"]

        inputs = dict()
        for idx in range(len(batch)):
            inputs[input_args[idx]] = (
                batch[idx]
                .view(
                    self.config.data.batch_size
                    * (self.config.data.num_neg + 1)
                    * self.config.data.overflow_limit,
                    -1,
                )
                .to(self.config.device)
            )

        outputs = self.encoder(**inputs)

        outputs = outputs.view(
            self.config.data.batch_size,
            self.config.data.num_neg + 1,
            self.config.data.overflow_limit,
        )
        mean_outputs = torch.mean(outputs, dim=-1)
        similarity_scores = F.log_softmax(mean_outputs, dim=-1)
        loss = self.criterion(similarity_scores, targets)
        self.log("step_train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch):
        targets = torch.zeros(self.config.data.batch_size).long().to(self.config.device)
        input_args = ["input_ids", "attention_mask", "token_type_ids"]

        inputs = dict()
        for idx in range(len(batch)):
            inputs[input_args[idx]] = (
                batch[idx]
                .view(
                    self.config.data.batch_size
                    * (self.config.data.num_neg + 1)
                    * self.config.data.overflow_limit,
                    -1,
                )
                .to(self.config.device)
            )

        outputs = self.encoder(**inputs)

        outputs = outputs.view(
            self.config.data.batch_size,
            self.config.data.num_neg + 1,
            self.config.data.overflow_limit,
        )

        mean_outputs = torch.mean(outputs, dim=-1)
        similarity_scores = F.log_softmax(mean_outputs, dim=-1)

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
        self.encoder.eval()
        self.dense_emb_matrix = None
        self.contexts = None
        pass

    def search(self, query, k=1):
        self.encoder.eval()

        tokenized_query = self.tokenizer(
            [query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.config.data.max_seq_length,
        )
        query_vector = self.encoder(**tokenized_query)

        similarity_score = torch.matmul(query_vector, self.dense_emb_matrix).squeeze()
        sorted_idx = torch.argsort(similarity_score, dim=-1, descending=True).squeeze()
        doc_scores = similarity_score[sorted_idx]

        return doc_scores[:k], self.contexts[sorted_idx[:k]]
