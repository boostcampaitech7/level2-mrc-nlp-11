import os, json
from IPython.display import display, HTML
from utils.data_template import get_dataset_list
from datasets import concatenate_datasets
import numpy as np


class SparseRetrievalAnalysis:

    def __init__(self, retrieval_model1=None, retrieval_model2=None, dataset_name=None):
        self.retrieval_model1 = retrieval_model1
        self.retrieval_model2 = retrieval_model2
        self.dataset_name = dataset_name
        self.result_list = None

    def load_result(self, result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as f:
            self.result_list = json.load(f)

    def calculate_result(self):
        dataset_list = get_dataset_list(self.dataset_name)
        self.eval_examples = concatenate_datasets(
            [ds["validation"] for ds in dataset_list]
        )
        self.correct_docs_idx, self.contexts = self.get_correct_docs_idx()

        _, self.retrieval1_docs_idx, _, self.retrieval1_query_score_list = (
            self.retrieval_model1.search(
                self.eval_examples["question"], k=1, return_query_score=True
            )
        )

    def calculate_compare_result(self):
        dataset_list = get_dataset_list(self.dataset_name)
        self.eval_examples = concatenate_datasets(
            [ds["validation"] for ds in dataset_list]
        )
        self.correct_docs_idx, self.contexts = self.get_correct_docs_idx()

        _, self.retrieval1_docs_idx, _, self.retrieval1_query_score_list = (
            self.retrieval_model1.search(
                self.eval_examples["question"], k=1, return_query_score=True
            )
        )
        _, self.retrieval2_docs_idx, _, self.retrieval2_query_score_list = (
            self.retrieval_model2.search(
                self.eval_examples["question"], k=1, return_query_score=True
            )
        )

    def get_correct_docs_idx(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = (
            f"{parent_directory}/{self.retrieval_model1.config.data_path}"
        )

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        contexts = np.array(list(dict.fromkeys([v["text"] for v in data.values()])))

        if os.path.isfile("../correct_docs_idx.json"):
            with open("../correct_docs_idx.json", "r") as file:
                correct_docs_idx = json.load(file)
            return correct_docs_idx, contexts

    def save_result(self):
        result_list = []

        for idx in range(len(self.eval_examples)):
            result = {}
            result["question"] = self.eval_examples[idx]["question"]
            result["answer"] = self.eval_examples[idx]["answers"]["text"]
            result["answer-context"] = self.eval_examples[idx]["context"]
            result["answer-context_retrieval1-values"] = sorted(
                [
                    (key, float(value[self.correct_docs_idx[idx]]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval1-predict-context"] = list(
                self.contexts[self.retrieval1_docs_idx[idx]]
            )
            result["retrieval1-predict-context_retrieval1-values"] = sorted(
                [
                    (key, list(value[self.retrieval1_docs_idx[idx]]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval1_is_correct"] = (
                True
                if result["answer-context"]
                .replace(" ", "")
                .replace("\n", "")
                .replace("\\n", "")
                in [
                    predict_context.replace(" ", "")
                    .replace("\n", "")
                    .replace("\\n", "")
                    for predict_context in result["retrieval1-predict-context"]
                ]
                else False
            )
            result_list.append(result)

        retrieval1_name = self.retrieval_model1.__class__.__name__
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_data_path = f"{parent_directory}/data/{retrieval1_name}-result.json"
        with open(result_data_path, "w", encoding="utf-8") as json_file:
            json.dump(result_list, json_file, ensure_ascii=False, indent=4)

    def save_compare_result(self):
        result_list = []

        for idx in range(len(self.eval_examples)):
            result = {}
            result["question"] = self.eval_examples[idx]["question"]
            result["answer"] = self.eval_examples[idx]["answers"]["text"]
            result["answer-context"] = self.eval_examples[idx]["context"]
            result["answer-context_retrieval1-values"] = sorted(
                [
                    (key, float(value[self.correct_docs_idx[idx]]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["answer-context_retrieval2-values"] = sorted(
                [
                    (key, float(value[self.correct_docs_idx[idx]]))
                    for key, value in self.retrieval2_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval1-predict-context"] = list(
                self.contexts[self.retrieval1_docs_idx[idx]]
            )
            result["retrieval1-predict-context_retrieval1-values"] = sorted(
                [
                    (key, list(value[self.retrieval1_docs_idx[idx]]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["retrieval1-predict-context_retrieval2-values"] = sorted(
                [
                    (key, list(value[self.retrieval1_docs_idx[idx]]))
                    for key, value in self.retrieval2_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval2-predict-context"] = list(
                self.contexts[self.retrieval2_docs_idx[idx]]
            )
            result["retrieval2-predict-context_retrieval1-values"] = sorted(
                [
                    (key, list(value[self.retrieval2_docs_idx[idx]]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["retrieval2-predict-context_retrieval2-values"] = sorted(
                [
                    (key, list(value[self.retrieval2_docs_idx[idx]]))
                    for key, value in self.retrieval2_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval1_is_correct"] = (
                True
                if result["answer-context"]
                .replace(" ", "")
                .replace("\n", "")
                .replace("\\n", "")
                in [
                    predict_context.replace(" ", "")
                    .replace("\n", "")
                    .replace("\\n", "")
                    for predict_context in result["retrieval1-predict-context"]
                ]
                else False
            )
            result["retrieval2_is_correct"] = (
                True
                if result["answer-context"]
                .replace(" ", "")
                .replace("\n", "")
                .replace("\\n", "")
                in [
                    predict_context.replace(" ", "")
                    .replace("\n", "")
                    .replace("\\n", "")
                    for predict_context in result["retrieval2-predict-context"]
                ]
                else False
            )
            result_list.append(result)

        retrieval1_name = self.retrieval_model1.__class__.__name__
        retrieval2_name = self.retrieval_model2.__class__.__name__
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_data_path = f"{parent_directory}/data/{retrieval1_name}-{retrieval2_name}-compare-result.json"
        with open(result_data_path, "w", encoding="utf-8") as json_file:
            json.dump(result_list, json_file, ensure_ascii=False, indent=4)

    def get_only_retrieval1_correct_query_idx(self):
        only_retrieval1_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if result["retrieval1_is_correct"] and not result["retrieval2_is_correct"]:
                only_retrieval1_correct_query_idx.append(idx)
        return only_retrieval1_correct_query_idx

    def get_only_retrieval2_correct_query_idx(self):
        only_retrieval2_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if not result["retrieval1_is_correct"] and result["retrieval2_is_correct"]:
                only_retrieval2_correct_query_idx.append(idx)
        return only_retrieval2_correct_query_idx

    def get_both_incorrect_query_idx(self):
        both_incorrect_query_idx = []
        for idx, result in enumerate(self.result_list):
            if (
                not result["retrieval1_is_correct"]
                and not result["retrieval2_is_correct"]
            ):
                both_incorrect_query_idx.append(idx)
        return both_incorrect_query_idx

    def get_retrieval1_correct_query_idx(self):
        retrieval1_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if result["retrieval1_is_correct"]:
                retrieval1_correct_query_idx.append(idx)
        return retrieval1_correct_query_idx

    def get_retrieval1_incorrect_query_idx(self):
        retrieval1_incorrect_query_idx = []
        for idx, result in enumerate(self.result_list):
            if not result["retrieval1_is_correct"]:
                retrieval1_incorrect_query_idx.append(idx)
        return retrieval1_incorrect_query_idx

    def print_query_result(self, idx):
        result = self.result_list[idx]
        print("=" * 20)
        print("QUESTION: ")
        display(HTML(result["question"]))
        print("=" * 20)

        display("=" * 20)
        print(f"RETRIEVAL1 PREDICT CONTEXT: {result['retrieval1_is_correct']}")
        display(HTML(result["retrieval1-predict-context"][0]))
        print("-" * 20)
        print("RETRIEVAL1 VALUE OF RETRIEVAL1 PREDICT CONTEXT: ")
        print(
            f"{self.simplify_data(result['retrieval1-predict-context_retrieval1-values'])}"
        )
        print("=" * 20)

        display("=" * 20)
        print("ANSWER CONTEXT: ")
        display(HTML(result["answer-context"]))
        print("-" * 20)
        print("RETRIEVAL1 VALUE OF ANSWER CONTEXT: ")
        print(self.simplify_data(result["answer-context_retrieval1-values"]))
        print("=" * 20)
        print()
        print("\n\n\n")

    def print_compare_query_result(self, idx):
        result = self.result_list[idx]
        print("=" * 20)
        print("QUESTION: ")
        display(HTML(result["question"]))
        print("=" * 20)

        display("=" * 20)
        print(f"RETRIEVAL1 PREDICT CONTEXT: {result['retrieval1_is_correct']}")
        display(HTML(result["retrieval1-predict-context"][0]))
        print("-" * 20)
        print("RETRIEVAL1 VALUE OF RETRIEVAL1 PREDICT CONTEXT: ")
        print(
            f"{self.simplify_data(result['retrieval1-predict-context_retrieval1-values'])}"
        )
        print("-" * 20)
        print("RETRIEVAL2 VALUE OF RETRIEVAL1 PREDICT CONTEXT: ")
        print(
            self.simplify_data(result["retrieval1-predict-context_retrieval2-values"])
        )
        print("=" * 20)

        display("=" * 20)
        print(f"RETRIEVAL2 PREDICT CONTEXT: {result['retrieval2_is_correct']}")
        display(HTML(result["retrieval2-predict-context"][0]))
        print("-" * 20)
        print("RETRIEVAL1 VALUE OF RETRIEVAL2 PREDICT CONTEXT: ")
        print(
            self.simplify_data(result["retrieval2-predict-context_retrieval1-values"])
        )
        print("-" * 20)
        print("RETRIEVAL2 VALUE OF RETRIEVAL2 PREDICT CONTEXT: ")
        print(
            self.simplify_data(result["retrieval2-predict-context_retrieval2-values"])
        )
        print("=" * 20)

        display("=" * 20)
        print("ANSWER CONTEXT: ")
        display(HTML(result["answer-context"]))
        print("-" * 20)
        print("RETRIEVAL1 VALUE OF ANSWER CONTEXT: ")
        print(self.simplify_data(result["answer-context_retrieval1-values"]))
        print("-" * 20)
        print("RETRIEVAL2 VALUE OF ANSWER CONTEXT: ")
        print(self.simplify_data(result["answer-context_retrieval2-values"]))
        print("=" * 20)
        print()
        print("\n\n\n")

    def simplify_data(self, _list):
        sorted_by_value = sorted(
            _list, key=lambda x: -x[1][0] if isinstance(x[1], list) else -x[1]
        )
        only_top_1_value = [
            (token, value[0]) if isinstance(value, list) else (token, value)
            for token, value in sorted_by_value
        ]
        total_value = sum([value for token, value in only_top_1_value])
        percent_value = [
            (token, round((value / total_value) * 100, 1)) if total_value > 0 else 0
            for token, value in only_top_1_value
        ]

        return percent_value
