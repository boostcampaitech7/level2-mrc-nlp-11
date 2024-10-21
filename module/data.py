from abc import abstractmethod
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, default_data_collator
from datasets import concatenate_datasets, DatasetDict

from utils.data_template import get_dataset_list
import utils.preprocessing as preproc_module


class MrcDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)
        # tokenizer에 스페셜 토큰 추가
        if len(self.config.data.add_special_token) != 0:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(self.config.data.add_special_token)}
            )
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.train_examples = None
        self.eval_examples = None
        self.test_examples = None

    def setup(self, stage="fit"):

        dataset_list = get_dataset_list(self.config.data.dataset_name)

        if stage == "fit":
            datasets = DatasetDict()
            for split in ["train", "validation"]:
                datasets[split] = concatenate_datasets(
                    [ds[split] for ds in dataset_list]
                )
            self.train_examples = datasets["train"]
            self.eval_examples = datasets["validation"]
            self.train_dataset, self.train_examples = self.get_dataset(
                self.train_examples, self.prepare_train_features
            )
            self.eval_dataset, self.eval_examples = self.get_dataset(
                self.eval_examples, self.prepare_validation_features
            )
            print(self.train_dataset)
            print(self.eval_dataset)

        if stage == "test":
            self.test_examples = concatenate_datasets(
                [ds["test"] for ds in dataset_list]
            )
            self.test_dataset, self.test_examples = self.get_dataset(
                self.test_examples, self.prepare_validation_features
            )
            print(self.test_dataset)

    def preprocessing(self, examples):
        for preproc in self.config.data.preproc_list:
            examples = examples.map(getattr(preproc_module, preproc))
        return examples

    def get_dataset(self, examples, feat_func=None):
        # 전처리하는 단계
        examples = self.preprocessing(examples)

        # 모델 입력으로 들어갈 수 있게 변경한 단계
        dataset = examples.map(
            (self.prepare_validation_features if not feat_func else feat_func),
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers,
            remove_columns=examples.column_names,
        )
        return dataset, examples

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset.remove_columns(self.config.data.remove_columns),
            collate_fn=default_data_collator,
            batch_size=self.config.data.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset.remove_columns(
                ["offset_mapping"] + self.config.data.remove_columns
            ),
            collate_fn=default_data_collator,
            batch_size=self.config.data.batch_size,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset.remove_columns(
                ["offset_mapping"] + self.config.data.remove_columns
            ),
            collate_fn=default_data_collator,
            batch_size=self.config.data.batch_size,
        )

    def prepare_train_features(self, examples):
        """
        Args:
            examples: {'id', 'question', 'context', 'answers': {'answer_start', 'text'}}
        Returns:
            tokenized_examples: {'input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'}
        """
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.config.data.max_seq_length,
            stride=self.config.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["example_id"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        """
        Args:
            examples: {'id', 'question', 'context', 'answers': {'answer_start', 'text'}}
        Returns:
            tokenized_examples: {'input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id'}
        """
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.config.data.max_seq_length,
            stride=self.config.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            # tokenized_examples["example_id"].append(examples["id"])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples


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

    def use_overflow_token(self, p_with_neg):

        pad_token_id = self.tokenizer.pad_token_id
        overflow_tokenized_p_with_neg = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }
        overflow_size = []

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
            overflow_size.append(cnt_overflow)
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
        return overflow_tokenized_p_with_neg, overflow_size

    def cut_overflow_token(self, p_with_neg):

        truncate_tokenized_p_with_neg = self.tokenizer(
            p_with_neg,
            truncation=True,
            max_length=self.config.data.max_seq_length,
            padding="max_length",
        )
        overflow_size = [1] * len(truncate_tokenized_p_with_neg)

        return truncate_tokenized_p_with_neg, overflow_size


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
            tokenized_p_with_neg, overflow_size = (
                self.preprocess_module.use_overflow_token(p_with_neg)
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
            tokenized_p_with_neg, overflow_size = (
                self.preprocess_module.cut_overflow_token(p_with_neg)
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
        overflow_size = torch.tensor(overflow_size).view(
            -1, self.config.data.num_neg + 1
        )

        return TensorDataset(
            tokenized_p_with_neg["input_ids"],
            tokenized_p_with_neg["attention_mask"],
            tokenized_p_with_neg["token_type_ids"],
            tokenized_questions["input_ids"],
            tokenized_questions["attention_mask"],
            tokenized_questions["token_type_ids"],
            overflow_size,
        )
