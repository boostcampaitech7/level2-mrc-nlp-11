from abc import abstractmethod
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets, DatasetDict
from utils.data_template import get_dataset_list


class RetrievalDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def setup(self, stage="fit"):

        dataset_list = get_dataset_list(self.config.data.dataset_name)

        if stage == "fit":
            datasets = DatasetDict()
            for split in ["train", "validation"]:
                datasets[split] = concatenate_datasets(
                    [ds[split] for ds in dataset_list]
                )

            train_dataset = datasets["train"]
            eval_dataset = datasets["validation"]

            self.train_dataset = self.preprocessing(train_dataset)
            self.eval_dataset = self.preprocessing(eval_dataset)

        if stage == "test":
            test_dataset = concatenate_datasets([ds["test"] for ds in dataset_list])
            self.test_dataset = self.preprocessing(test_dataset)

    def random_neg_sampling(self, examples):
        corpus = list(set([example["context"] for example in examples]))
        p_with_neg = []

        for context, answers in zip(examples["context"], examples["answers"]):
            p_with_neg.append(context)
            cnt_neg = 0
            while cnt_neg < self.config.data.num_neg:
                neg_idx = random.randrange(0, len(corpus))
                # 만약 정답이 문서에 있을 경우 사용하지 않음
                if corpus[neg_idx] != context and not any(
                    text in corpus[neg_idx] for text in answers["text"]
                ):
                    p_with_neg.append(corpus[neg_idx])
                    cnt_neg += 1

        return p_with_neg

    def sparse_neg_sampling(self, examples):
        p_with_neg = []
        for context, negative_sample in zip(
            examples["context"], examples["negative_sample"]
        ):
            p_with_neg.append(context)
            p_with_neg.extend(negative_sample[: self.config.data.num_neg])

        return p_with_neg

    def combine_neg_sampling(self, examples):
        corpus = list(set([example["context"] for example in examples]))
        p_with_neg = []
        for context, negative_sample, answers in zip(
            examples["context"], examples["negative_sample"], examples["answers"]
        ):
            p_with_neg.append(context)
            p_with_neg.extend(negative_sample[: self.config.data.num_neg // 2])
            cnt_neg = 0
            while cnt_neg < self.config.data.num_neg // 2:
                neg_idx = random.randrange(0, len(corpus))
                # 만약 정답이 문서에 있을 경우 사용하지 않음
                if corpus[neg_idx] != context:
                    p_with_neg.append(corpus[neg_idx])
                    cnt_neg += 1

        return p_with_neg

    @abstractmethod
    def preprocessing(self, examples):
        assert NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.data.batch_size,
            drop_last=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            drop_last=True,
        )


class BiEncoderRetrievalPreprocDataModule:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)

    def process_overflow_token(self, p_with_neg):

        pad_token_id = self.tokenizer.pad_token_id
        overflow_tokenized_p_with_neg = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }

        tokenized_p_with_neg = self.tokenizer(
            p_with_neg,
            truncation=True,
            max_length=self.config.data.max_seq_length,
            stride=self.config.data.doc_stride,
            return_overflowing_tokens=True,
            padding="max_length",
        )

        sample_mapping = tokenized_p_with_neg.pop("overflow_to_sample_mapping")

        sample_idx, example_idx = -1, 0
        while len(sample_mapping) > example_idx:
            cnt_overflow = 0
            sample_idx += 1
            while (
                cnt_overflow < self.config.data.overflow_limit
                and len(sample_mapping) > example_idx
                and sample_idx == sample_mapping[example_idx]
            ):
                for key, value in tokenized_p_with_neg.items():
                    overflow_tokenized_p_with_neg[key].append(value[example_idx])
                cnt_overflow += 1
                example_idx += 1

            while cnt_overflow < self.config.data.overflow_limit:
                overflow_tokenized_p_with_neg["input_ids"].append(
                    [pad_token_id] * self.config.data.max_seq_length
                )
                overflow_tokenized_p_with_neg["attention_mask"].append(
                    [0] * self.config.data.max_seq_length
                )
                overflow_tokenized_p_with_neg["token_type_ids"].append(
                    [0] * self.config.data.max_seq_length
                )
                cnt_overflow += 1

            while (
                len(sample_mapping) > example_idx
                and sample_mapping[example_idx] == sample_idx
            ):
                example_idx += 1
        return overflow_tokenized_p_with_neg

    def truncate_overflow_token(self, p_with_neg):

        truncate_tokenized_p_with_neg = self.tokenizer(
            p_with_neg,
            truncation=True,
            max_length=self.config.data.max_seq_length,
            padding="max_length",
        )
        return truncate_tokenized_p_with_neg


class CrossEncoderRetrievalPreprocDataModule:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)

    def process_overflow_token(self, questions, p_with_neg):
        pad_token_id = self.tokenizer.pad_token_id
        overflow_tokenized_q_p = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }
        for question_idx, question in enumerate(questions):
            for i in range(self.config.data.num_neg + 1):
                tokenized_q_p = self.tokenizer(
                    question,
                    p_with_neg[question_idx * (self.config.data.num_neg + 1) + i],
                    truncation="only_second",
                    max_length=self.config.data.max_seq_length,
                    stride=self.config.data.doc_stride,
                    return_overflowing_tokens=True,
                    padding="max_length",
                )

                sample_mapping = tokenized_q_p.pop("overflow_to_sample_mapping")

                cnt_overflow = 0
                while (
                    cnt_overflow < len(sample_mapping)
                    and cnt_overflow < self.config.data.overflow_limit
                ):
                    for key, value in tokenized_q_p.items():
                        overflow_tokenized_q_p[key].append(value[cnt_overflow])
                    cnt_overflow += 1

                while cnt_overflow < self.config.data.overflow_limit:
                    overflow_tokenized_q_p["input_ids"].append(
                        [pad_token_id] * self.config.data.max_seq_length
                    )
                    overflow_tokenized_q_p["attention_mask"].append(
                        [0] * self.config.data.max_seq_length
                    )
                    overflow_tokenized_q_p["token_type_ids"].append(
                        [0] * self.config.data.max_seq_length
                    )
                    cnt_overflow += 1

        return overflow_tokenized_q_p

    def truncate_overflow_token(self, questions, p_with_neg):
        truncate_tokenized_q_p = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }

        for question_idx, question in enumerate(questions):
            for i in range(self.config.data.num_neg + 1):
                tokenized_q_p = self.tokenizer(
                    question,
                    p_with_neg[question_idx * (self.config.data.num_neg + 1) + i],
                    truncation=True,
                    max_length=self.config.data.max_seq_length,
                    padding="max_length",
                )
                for key, value in tokenized_q_p.items():
                    truncate_tokenized_q_p[key].append(value)

        return truncate_tokenized_q_p


class BiEncoderRetrievalDataModule(RetrievalDataModule):

    def __init__(self, config):
        super().__init__(config)
        self.preprocess_module = BiEncoderRetrievalPreprocDataModule(config)

    def preprocessing(self, examples):
        tokenized_questions = self.preprocess_module.tokenizer(
            examples["question"],
            truncation=True,
            max_length=self.config.data.max_seq_length,
            return_tensors="pt",
            padding="max_length",
        )

        p_with_neg = getattr(self, self.config.data.neg_sampling_method)(examples)
        if self.config.data.use_overflow_token:
            tokenized_p_with_neg = self.preprocess_module.process_overflow_token(
                p_with_neg
            )
            tokenized_p_with_neg["input_ids"] = torch.tensor(
                tokenized_p_with_neg["input_ids"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.overflow_limit,
                self.config.data.max_seq_length,
            )
            tokenized_p_with_neg["attention_mask"] = torch.tensor(
                tokenized_p_with_neg["attention_mask"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.overflow_limit,
                self.config.data.max_seq_length,
            )
            tokenized_p_with_neg["token_type_ids"] = torch.tensor(
                tokenized_p_with_neg["token_type_ids"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.overflow_limit,
                self.config.data.max_seq_length,
            )
        else:
            tokenized_p_with_neg = self.preprocess_module.truncate_overflow_token(
                p_with_neg
            )
            tokenized_p_with_neg["input_ids"] = torch.tensor(
                tokenized_p_with_neg["input_ids"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.max_seq_length,
            )
            tokenized_p_with_neg["attention_mask"] = torch.tensor(
                tokenized_p_with_neg["attention_mask"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.max_seq_length,
            )
            tokenized_p_with_neg["token_type_ids"] = torch.tensor(
                tokenized_p_with_neg["token_type_ids"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.max_seq_length,
            )

        return TensorDataset(
            tokenized_p_with_neg["input_ids"],
            tokenized_p_with_neg["attention_mask"],
            tokenized_p_with_neg["token_type_ids"],
            tokenized_questions["input_ids"],
            tokenized_questions["attention_mask"],
            tokenized_questions["token_type_ids"],
        )


class CrossEncoderRetrievalDataModule(RetrievalDataModule):

    def __init__(self, config):
        super().__init__(config)
        self.preprocess_module = CrossEncoderRetrievalPreprocDataModule(config)

    def preprocessing(self, examples):
        p_with_neg = getattr(self, self.config.data.neg_sampling_method)(examples)

        if self.config.data.use_overflow_token:
            tokenized_q_p = self.preprocess_module.process_overflow_token(
                examples["question"], p_with_neg
            )
            tokenized_q_p["input_ids"] = torch.tensor(tokenized_q_p["input_ids"]).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.overflow_limit,
                self.config.data.max_seq_length,
            )
            tokenized_q_p["attention_mask"] = torch.tensor(
                tokenized_q_p["attention_mask"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.overflow_limit,
                self.config.data.max_seq_length,
            )
            tokenized_q_p["token_type_ids"] = torch.tensor(
                tokenized_q_p["token_type_ids"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.overflow_limit,
                self.config.data.max_seq_length,
            )
        else:
            tokenized_q_p = self.preprocess_module.truncate_overflow_token(
                examples["question"], p_with_neg
            )
            tokenized_q_p["input_ids"] = torch.tensor(tokenized_q_p["input_ids"]).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.max_seq_length,
            )
            tokenized_q_p["attention_mask"] = torch.tensor(
                tokenized_q_p["attention_mask"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.max_seq_length,
            )
            tokenized_q_p["token_type_ids"] = torch.tensor(
                tokenized_q_p["token_type_ids"]
            ).view(
                -1,
                self.config.data.num_neg + 1,
                self.config.data.max_seq_length,
            )

        if any(
            model_type in self.config.model.plm_name
            for model_type in ["roberta", "Roberta"]
        ):
            return TensorDataset(
                tokenized_q_p["input_ids"],
                tokenized_q_p["attention_mask"],
            )
        return TensorDataset(
            tokenized_q_p["input_ids"],
            tokenized_q_p["attention_mask"],
            tokenized_q_p["token_type_ids"],
        )
