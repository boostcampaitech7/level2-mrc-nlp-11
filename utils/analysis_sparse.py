import os, json
from IPython.display import display, HTML
from module.retrieval import Bm25Retrieval, TfIdfRetrieval
from utils.data_template import get_dataset_list
from datasets import concatenate_datasets
import numpy as np


class SparseRetrievalAnalysis:

    def __init__(self, config):
        self.config = config
        self.result_list = None

    def load_result(self, result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as f:
            self.result_list = json.load(f)

    def calculate_result(self):
        self.tfidf = TfIdfRetrieval(self.config)
        self.bm25 = Bm25Retrieval(self.config)
        self.tfidf.fit()
        self.bm25.fit()

        dataset_list = get_dataset_list(self.config.data.dataset_name)
        self.eval_examples = concatenate_datasets(
            [ds["validation"] for ds in dataset_list]
        )
        self.correct_docs_idx, self.contexts = self.get_correct_docs_idx()
        print("get correct context")
        _, _, self.tfidf_docs_idx, self.tfidf_q_scores = self.tfidf.search(
            self.eval_examples["question"], k=1
        )
        print("get tfidf docs idx, q scores")
        _, _, self.bm25_docs_idx, self.bm25_q_scores = self.bm25.search(
            self.eval_examples["question"], k=1
        )
        print("get bm25 docs idx, q scores")

    def get_correct_docs_idx(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        context_data_path = f"{current_directory}/{self.config.tfidf.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        contexts = np.array(list(dict.fromkeys([v["text"] for v in data.values()])))

        if os.path.isfile("../correct_docs_idx.json"):
            with open("../correct_docs_idx.json", "r") as file:
                correct_docs_idx = json.load(file)
            return correct_docs_idx, contexts

        correct_docs_idx = []
        for example in self.eval_examples:
            idx = -1
            for i, context in enumerate(contexts):
                if example["context"] == context:
                    idx = i
                    break
            correct_docs_idx.append(idx)
            assert idx != -1

        with open("../correct_docs_idx.json", "w") as file:
            json.dump(correct_docs_idx, file)
        return correct_docs_idx, contexts

    def save_result(self):
        result_list = []
        for idx in range(len(self.eval_examples)):
            result = {}
            result["question"] = self.eval_examples[idx]["question"]
            result["answer-context"] = self.eval_examples[idx]["context"]
            result["answer-context_tfidf-values"] = sorted(
                [
                    (key, float(value[self.correct_docs_idx[idx]]))
                    for key, value in self.tfidf_q_scores[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["answer-context_bm25-values"] = sorted(
                [
                    (key, float(value[self.correct_docs_idx[idx]]))
                    for key, value in self.bm25_q_scores[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["answer-context_value-diff"] = [
                (tfidf_k, tfidf_v - bm25_v)
                for (tfidf_k, tfidf_v), (bm25_k, bm25_v) in zip(
                    result["answer-context_tfidf-values"],
                    result["answer-context_bm25-values"],
                )
            ]

            result["tfidf-predict-context"] = list(
                self.contexts[self.tfidf_docs_idx[idx]]
            )
            result["tfidf-predict-context_tfidf-values"] = sorted(
                [
                    (key, list(value[self.tfidf_docs_idx[idx]]))
                    for key, value in self.tfidf_q_scores[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["tfidf-predict-context_bm25-values"] = sorted(
                [
                    (key, list(value[self.tfidf_docs_idx[idx]]))
                    for key, value in self.bm25_q_scores[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["tfidf-predict-context_value-diff"] = [
                (tfidf_k, [t - b for t, b in zip(tfidf_v, bm25_v)])
                for (tfidf_k, tfidf_v), (bm25_k, bm25_v) in zip(
                    result["tfidf-predict-context_tfidf-values"],
                    result["tfidf-predict-context_bm25-values"],
                )
            ]

            result["bm25-predict-context"] = list(
                self.contexts[self.bm25_docs_idx[idx]]
            )
            result["bm25-predict-context_tfidf-values"] = sorted(
                [
                    (key, list(value[self.bm25_docs_idx[idx]]))
                    for key, value in self.tfidf_q_scores[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["bm25-predict-context_bm25-values"] = sorted(
                [
                    (key, list(value[self.bm25_docs_idx[idx]]))
                    for key, value in self.bm25_q_scores[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["bm25-predict-context_value-diff"] = [
                (tfidf_k, [t - b for t, b in zip(tfidf_v, bm25_v)])
                for (tfidf_k, tfidf_v), (bm25_k, bm25_v) in zip(
                    result["bm25-predict-context_tfidf-values"],
                    result["bm25-predict-context_bm25-values"],
                )
            ]

            result["tfidf_is_correct"] = (
                True
                if result["answer-context"]
                .replace(" ", "")
                .replace("\n", "")
                .replace("\\n", "")
                in [
                    predict_context.replace(" ", "")
                    .replace("\n", "")
                    .replace("\\n", "")
                    for predict_context in result["tfidf-predict-context"]
                ]
                else False
            )
            result["bm25_is_correct"] = (
                True
                if result["answer-context"]
                .replace(" ", "")
                .replace("\n", "")
                .replace("\\n", "")
                in [
                    predict_context.replace(" ", "")
                    .replace("\n", "")
                    .replace("\\n", "")
                    for predict_context in result["bm25-predict-context"]
                ]
                else False
            )
            result_list.append(result)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        result_data_path = f"{current_directory}/data/sparse_retrieval_compare.json"
        with open(result_data_path, "w", encoding="utf-8") as json_file:
            json.dump(result_list, json_file, ensure_ascii=False, indent=4)

    def get_bm25_correct_query_idx(self):
        only_bm25_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if result["bm25_is_correct"] and not result["tfidf_is_correct"]:
                only_bm25_correct_query_idx.append(idx)
        return only_bm25_correct_query_idx

    def get_tfidf_correct_query_idx(self):
        only_tfidf_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if not result["bm25_is_correct"] and result["tfidf_is_correct"]:
                only_tfidf_correct_query_idx.append(idx)
        return only_tfidf_correct_query_idx

    def get_both_incorrect_query_idx(self):
        both_incorrect_q_idx = []
        for idx, result in enumerate(self.result_list):
            if not result["bm25_is_correct"] and not result["tfidf_is_correct"]:
                both_incorrect_q_idx.append(idx)
        return both_incorrect_q_idx

    def print_query_idx_result(self, idx):
        result = self.result_list[idx]
        print("=" * 20)
        print("QUESTION: ")
        display(HTML(result["question"]))
        print("=" * 20)

        display("=" * 20)
        print(f"TF-IDF PREDICT CONTEXT: {result['tfidf_is_correct']}")
        display(HTML(result["tfidf-predict-context"][0]))
        print("-" * 20)
        print("TF-IDF VALUE OF TF-IDF PREDICT CONTEXT: ")
        print(f"{self.simplify_data(result['tfidf-predict-context_tfidf-values'])}")
        print("-" * 20)
        print("BM25 VALUE OF TF-IDF PREDICT CONTEXT: ")
        print(self.simplify_data(result["tfidf-predict-context_bm25-values"]))
        print("=" * 20)

        display("=" * 20)
        print(f"BM25 PREDICT CONTEXT: {result['bm25_is_correct']}")
        display(HTML(result["bm25-predict-context"][0]))
        print("-" * 20)
        print("TF-IDF VALUE OF BM25 PREDICT CONTEXT: ")
        print(self.simplify_data(result["bm25-predict-context_tfidf-values"]))
        print("-" * 20)
        print("BM25 VALUE OF BM25 PREDICT CONTEXT: ")
        print(self.simplify_data(result["bm25-predict-context_bm25-values"]))
        print("=" * 20)

        display("=" * 20)
        print("ANSWER CONTEXT: ")
        display(HTML(result["answer-context"]))
        print("-" * 20)
        print("TF-IDF VALUE OF ANSWER CONTEXT: ")
        print(self.simplify_data(result["answer-context_tfidf-values"]))
        print("-" * 20)
        print("BM25 VALUE OF ANSWER CONTEXT: ")
        print(self.simplify_data(result["answer-context_bm25-values"]))
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
