import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import json
from datasets import Dataset, concatenate_datasets
import copy
from data_template import get_standard_features

load_dotenv()


def add_modified_question(examples, mode):
    modified_train_questions_path = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/data/paraphrased_train_questions.json"
    )
    modified_validation_questions_path = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/data/paraphrased_validation_questions.json"
    )

    if mode == "train":
        with open(modified_train_questions_path, "r", encoding="utf-8") as f:
            modified_questions = json.load(f)

    else:
        with open(modified_validation_questions_path, "r", encoding="utf-8") as f:
            modified_questions = json.load(f)

    modified_data = []
    for example in examples:
        if example["question"] in modified_questions:
            modified_example = copy.deepcopy(example)
            modified_example["question"] = modified_questions[example["question"]]
            modified_example["id"] = example["id"] + "_modified"
            modified_data.append(modified_example)

    if modified_data:
        modified_dataset = Dataset.from_list(
            modified_data, features=get_standard_features()
        )
        combined_dataset = concatenate_datasets([examples, modified_dataset])
        combined_dataset = combined_dataset.shuffle(seed=42)

    return combined_dataset


def modify_question(examples, mode):
    modified_train_questions_path = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/data/paraphrased_train_questions.json"
    )
    modified_validation_questions_path = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/data/paraphrased_validation_questions.json"
    )

    if mode == "train":
        with open(modified_train_questions_path, "r", encoding="utf-8") as f:
            modified_questions = json.load(f)

    else:
        with open(modified_validation_questions_path, "r", encoding="utf-8") as f:
            modified_questions = json.load(f)

    for example in examples:
        if example["question"] in modified_questions:
            example["question"] = modified_questions[example["question"]]

    return examples


if __name__ == "__main__":
    add_modified_question()
