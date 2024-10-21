import os, json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from evaluate import load
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig

import module.loss as module_loss
from utils.common import init_obj
import module.metric as module_metric
import module.dense_retrieval_encoder as module_encoder
from module.dense_retrieval_data import (
    BiEncoderRetrievalPreprocDataModule,
    CrossEncoderRetrievalPreprocDataModule,
)


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
                self.preprocess_module.truncate_overflow_token(self.contexts)
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
                    .view(
                        -1,
                        self.config.data.max_seq_length,
                    )
                    .to(self.config.device)
                    .long()
                )
                sub_tokenized_contexts["attention_mask"] = (
                    torch.tensor(
                        tokenized_contexts["attention_mask"][
                            offset * i : offset * (i + 1)
                        ]
                    )
                    .view(
                        -1,
                        self.config.data.max_seq_length,
                    )
                    .to(self.config.device)
                    .long()
                )
                sub_tokenized_contexts["token_type_ids"] = (
                    torch.tensor(
                        tokenized_contexts["token_type_ids"][
                            offset * i : offset * (i + 1)
                        ]
                    )
                    .view(
                        -1,
                        self.config.data.max_seq_length,
                    )
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
                doc_scores[:k],
                sorted_idx[:k],
                [self.contexts[idx] for idx in sorted_idx[:k]],
                similarity_score.cpu().numpy(),
            )

        return (
            doc_scores[:k],
            sorted_idx[:k],
            [self.contexts[idx] for idx in sorted_idx[:k]],
        )


class CrossEncoderDenseRetrieval(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.preprocess_module = CrossEncoderRetrievalPreprocDataModule(config)

        self.save_hyperparameters()
        self.encoder = module_encoder.CrossEncoder(self.config.model.plm_name).to(
            self.config.device
        )
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
                .view(-1, self.config.data.max_seq_length)
                .to(self.config.device)
            )

        outputs = self.encoder(inputs)

        outputs = outputs.view(
            self.config.data.batch_size,
            self.config.data.num_neg + 1,
            -1,
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
                .view(-1, self.config.data.max_seq_length)
                .to(self.config.device)
            )

        outputs = self.encoder(inputs)

        outputs = outputs.view(
            self.config.data.batch_size, self.config.data.num_neg + 1, -1
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

    def search(self, contexts, query, k=1):
        self.encoder.eval()

        self.preprocess_module.config.data.num_neg = len(contexts) - 1
        if not self.config.data.use_overflow_token:
            tokenized_q_p = self.preprocess_module.process_overflow_token(
                [query], contexts
            )
        else:
            tokenized_q_p = self.preprocess_module.truncate_overflow_token(
                [query], contexts
            )

        with torch.no_grad():
            tokenized_q_p["input_ids"] = torch.tensor(tokenized_q_p["input_ids"]).view(
                -1,
                self.config.data.max_seq_length,
            )
            tokenized_q_p["attention_mask"] = torch.tensor(
                tokenized_q_p["attention_mask"]
            ).view(
                -1,
                self.config.data.max_seq_length,
            )
            tokenized_q_p["token_type_ids"] = torch.tensor(
                tokenized_q_p["token_type_ids"]
            ).view(
                -1,
                self.config.data.max_seq_length,
            )

            outputs = self.encoder(tokenized_q_p)

        outputs = outputs.view(len(contexts) - 1, -1)
        similarity_score = torch.mean(outputs, dim=-1).squeeze()
        sorted_idx = torch.argsort(similarity_score, dim=-1, descending=True).squeeze()
        doc_scores = similarity_score[sorted_idx]

        return doc_scores[:k], [contexts[i] for i in sorted_idx[:k]]
