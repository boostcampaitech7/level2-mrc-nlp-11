import os, json
from IPython.display import display, HTML
from utils.data_template import get_dataset_list
from datasets import concatenate_datasets
import pandas as pd
import streamlit as st


class SparseRetrievalResultProvider:

    def __init__(self, retrieval_model1=None, retrieval_model2=None):
        self.retrieval_model1 = retrieval_model1
        self.retrieval_model2 = retrieval_model2

    def get_correct_docs_idx_key_pair(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isfile(f"{parent_directory}/data/correct_docs_idx_key_pair.json"):
            with open(
                f"{parent_directory}/data/correct_docs_idx_key_pair.json", "r"
            ) as file:
                correct_docs_idx_key_pair = json.load(file)

            return {int(key): value for key, value in correct_docs_idx_key_pair.items()}

        context_data_path = f"{parent_directory}/data/wikipedia_documents.json"
        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        text_key_pair = {v["text"]: k for k, v in data.items()}

        correct_docs_idx_key_pair = {}
        for idx, example in enumerate(self.eval_examples):
            for text, key in text_key_pair.items():
                if example["context"] == text:
                    correct_docs_idx_key_pair[idx] = key
                    break

        with open(
            f"{parent_directory}/data/correct_docs_idx_key_pair.json", "w"
        ) as file:
            json.dump(correct_docs_idx_key_pair, file)
        return correct_docs_idx_key_pair

    def calculate_result(self):
        dataset_list = get_dataset_list(["default"])
        self.eval_examples = concatenate_datasets(
            [ds["validation"] for ds in dataset_list]
        )
        self.correct_docs_idx_key_pair = self.get_correct_docs_idx_key_pair()

        (
            _,
            self.retrieval1_docs_idx,
            self.retrieval1_docs,
            self.retrieval1_query_score_list,
        ) = self.retrieval_model1.search(
            self.eval_examples["question"], k=1, return_query_score=True
        )

    def calculate_compare_result(self):
        dataset_list = get_dataset_list(["default"])
        self.eval_examples = concatenate_datasets(
            [ds["validation"] for ds in dataset_list]
        )
        self.correct_docs_idx_key_pair = self.get_correct_docs_idx_key_pair()

        (
            _,
            self.retrieval1_docs_idx,
            self.retrieval1_docs,
            self.retrieval1_query_score_list,
        ) = self.retrieval_model1.search(
            self.eval_examples["question"], k=1, return_query_score=True
        )
        (
            _,
            self.retrieval2_docs_idx,
            self.retrieval2_docs,
            self.retrieval2_query_score_list,
        ) = self.retrieval_model2.search(
            self.eval_examples["question"], k=1, return_query_score=True
        )

    def save_result(self):
        result_list = []

        retrieval1_contexts_key_idx_pair = self.retrieval_model1.contexts_key_idx_pair
        retrieval1_contexts_idx_key_pair = {
            v: k for k, v in retrieval1_contexts_key_idx_pair.items()
        }
        for idx in range(len(self.eval_examples)):
            correct_doc_key = self.correct_docs_idx_key_pair[idx]
            retrieval1_correct_doc_idx = retrieval1_contexts_key_idx_pair[
                correct_doc_key
            ]
            retrieval1_predict_doc_key = [
                retrieval1_contexts_idx_key_pair[doc_idx]
                for doc_idx in self.retrieval1_docs_idx[idx]
            ]
            retrieval1_predict_doc_idx = [
                retrieval1_contexts_key_idx_pair[doc_key]
                for doc_key in retrieval1_predict_doc_key
            ]

            result = {}
            result["question"] = self.eval_examples[idx]["question"]
            result["answer"] = self.eval_examples[idx]["answers"]["text"]
            result["answer-context"] = self.eval_examples[idx]["context"]
            result["answer-context_retrieval1-values"] = sorted(
                [
                    (key, float(value[retrieval1_correct_doc_idx]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval1-predict-context"] = self.retrieval1_docs[idx]
            result["retrieval1-predict-context_retrieval1-values"] = sorted(
                [
                    (key, list(value[retrieval1_predict_doc_idx]))
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
        result_data_path = (
            f"{parent_directory}/data/{retrieval1_name}-single-result.json"
        )
        with open(result_data_path, "w", encoding="utf-8") as json_file:
            json.dump(result_list, json_file, ensure_ascii=False, indent=4)

    def save_compare_result(self):
        result_list = []

        retrieval1_contexts_key_idx_pair = self.retrieval_model1.contexts_key_idx_pair
        retrieval2_contexts_key_idx_pair = self.retrieval_model2.contexts_key_idx_pair
        retrieval1_contexts_idx_key_pair = {
            v: k for k, v in retrieval1_contexts_key_idx_pair.items()
        }
        retrieval2_contexts_idx_key_pair = {
            v: k for k, v in retrieval2_contexts_key_idx_pair.items()
        }
        for idx in range(len(self.eval_examples)):
            correct_doc_key = self.correct_docs_idx_key_pair[idx]
            retrieval1_correct_doc_idx = retrieval1_contexts_key_idx_pair[
                correct_doc_key
            ]
            retrieval2_correct_doc_idx = retrieval2_contexts_key_idx_pair[
                correct_doc_key
            ]

            retrieval1_predict_doc_key = [
                retrieval1_contexts_idx_key_pair[doc_idx]
                for doc_idx in self.retrieval1_docs_idx[idx]
            ]
            retrieval1_predict1_doc_idx = [
                retrieval1_contexts_key_idx_pair[doc_key]
                for doc_key in retrieval1_predict_doc_key
            ]
            retrieval2_predict1_doc_idx = [
                retrieval2_contexts_key_idx_pair[doc_key]
                for doc_key in retrieval1_predict_doc_key
            ]

            retrieval2_predict_doc_key = [
                retrieval2_contexts_idx_key_pair[doc_idx]
                for doc_idx in self.retrieval2_docs_idx[idx]
            ]
            retrieval1_predict2_doc_idx = [
                retrieval1_contexts_key_idx_pair[doc_key]
                for doc_key in retrieval2_predict_doc_key
            ]
            retrieval2_predict2_doc_idx = [
                retrieval2_contexts_key_idx_pair[doc_key]
                for doc_key in retrieval2_predict_doc_key
            ]

            result = {}
            result["question"] = self.eval_examples[idx]["question"]
            result["answer"] = self.eval_examples[idx]["answers"]["text"]
            result["answer-context"] = self.eval_examples[idx]["context"]
            result["answer-context_retrieval1-values"] = sorted(
                [
                    (key, float(value[retrieval1_correct_doc_idx]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["answer-context_retrieval2-values"] = sorted(
                [
                    (key, float(value[retrieval2_correct_doc_idx]))
                    for key, value in self.retrieval2_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval1-predict-context"] = self.retrieval1_docs[idx]
            result["retrieval1-predict-context_retrieval1-values"] = sorted(
                [
                    (key, list(value[retrieval1_predict1_doc_idx]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["retrieval1-predict-context_retrieval2-values"] = sorted(
                [
                    (key, list(value[retrieval2_predict1_doc_idx]))
                    for key, value in self.retrieval2_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )

            result["retrieval2-predict-context"] = self.retrieval2_docs[idx]
            result["retrieval2-predict-context_retrieval1-values"] = sorted(
                [
                    (key, list(value[retrieval1_predict2_doc_idx]))
                    for key, value in self.retrieval1_query_score_list[idx].items()
                ],
                key=lambda x: x[0],
            )
            result["retrieval2-predict-context_retrieval2-values"] = sorted(
                [
                    (key, list(value[retrieval2_predict2_doc_idx]))
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


class SparseRetrievalResultViewer:

    def __init__(self, result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as f:
            self.result_list = json.load(f)
        self.type = result_file_path.split("/")[-1].split("-")[-2]
        if self.type == "single":
            self.result_method1 = result_file_path.split("/")[-1].split("-")[0]
            self.result_method2 = None
        elif self.type == "compare":
            self.result_method1 = result_file_path.split("/")[-1].split("-")[0]
            self.result_method2 = result_file_path.split("/")[-1].split("-")[1]
        else:
            self.result_method1 = None
            self.result_method2 = None

    def get_only_retrieval1_correct_query_idx(self):
        assert self.result_method1 != None and self.result_method2 != None
        only_retrieval1_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if result["retrieval1_is_correct"] and not result["retrieval2_is_correct"]:
                only_retrieval1_correct_query_idx.append(idx)
        return only_retrieval1_correct_query_idx

    def get_only_retrieval2_correct_query_idx(self):
        assert self.result_method1 != None and self.result_method2 != None
        only_retrieval2_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if not result["retrieval1_is_correct"] and result["retrieval2_is_correct"]:
                only_retrieval2_correct_query_idx.append(idx)
        return only_retrieval2_correct_query_idx

    def get_both_incorrect_query_idx(self):
        assert self.result_method1 != None and self.result_method2 != None
        both_incorrect_query_idx = []
        for idx, result in enumerate(self.result_list):
            if (
                not result["retrieval1_is_correct"]
                and not result["retrieval2_is_correct"]
            ):
                both_incorrect_query_idx.append(idx)
        return both_incorrect_query_idx

    def get_retrieval1_correct_query_idx(self):
        assert self.result_method1 != None
        retrieval1_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if result["retrieval1_is_correct"]:
                retrieval1_correct_query_idx.append(idx)
        return retrieval1_correct_query_idx

    def get_retrieval1_incorrect_query_idx(self):
        assert self.result_method1 != None
        retrieval1_incorrect_query_idx = []
        for idx, result in enumerate(self.result_list):
            if not result["retrieval1_is_correct"]:
                retrieval1_incorrect_query_idx.append(idx)
        return retrieval1_incorrect_query_idx

    def get_retrieval2_correct_query_idx(self):
        assert self.result_method2 != None
        retrieval2_correct_query_idx = []
        for idx, result in enumerate(self.result_list):
            if result["retrieval2_is_correct"]:
                retrieval2_correct_query_idx.append(idx)
        return retrieval2_correct_query_idx

    def get_retrieval2_incorrect_query_idx(self):
        assert self.result_method2 != None
        retrieval2_incorrect_query_idx = []
        for idx, result in enumerate(self.result_list):
            if not result["retrieval2_is_correct"]:
                retrieval2_incorrect_query_idx.append(idx)
        return retrieval2_incorrect_query_idx

    def simplify_data(self, _list, return_precent=False, return_total_value=True):
        sorted_by_value = sorted(
            _list, key=lambda x: -x[1][0] if isinstance(x[1], list) else -x[1]
        )
        only_top_1_value = [
            (
                (token, round(value[0], 3))
                if isinstance(value, list)
                else (token, round(value, 3))
            )
            for token, value in sorted_by_value
        ]
        total_value = sum([value for token, value in only_top_1_value])
        if not return_precent:
            return (
                [("총합", round(total_value, 3))] + only_top_1_value
                if return_total_value
                else only_top_1_value
            )

        percent_value = [
            (token, round((value / total_value) * 100, 1)) if total_value > 0 else 0
            for token, value in only_top_1_value
        ]
        return (
            [("총합", round(total_value, 3))] + percent_value
            if return_total_value
            else percent_value
        )

    def print_query_result(self, idx):
        assert self.result_method1 != None
        result = self.result_list[idx]
        print("=" * 20)
        print("QUESTION: ")
        display(HTML(result["question"]))
        print("=" * 20)

        display("=" * 20)
        print(
            f"{self.result_method1}-PREDICT-CONTEXT: {result['retrieval1_is_correct']}"
        )
        display(HTML(result["retrieval1-predict-context"][0]))
        print("-" * 20)
        print(f"{self.result_method1}-VALUE OF {self.result_method1}-PREDICT-CONTEXT: ")
        print(
            f"{self.simplify_data(result['retrieval1-predict-context_retrieval1-values'])}"
        )
        print("=" * 20)

        display("=" * 20)
        print("ANSWER-CONTEXT: ")
        display(HTML(result["answer-context"]))
        print("-" * 20)
        print(f"{self.result_method1}-VALUE OF ANSWER-CONTEXT: ")
        print(self.simplify_data(result["answer-context_retrieval1-values"]))
        print("=" * 20)
        print()
        print("\n\n\n")

    def print_compare_query_result(self, idx):
        assert self.result_method1 != None and self.result_method2 != None
        result = self.result_list[idx]
        print("=" * 20)
        print("QUESTION: ")
        display(HTML(result["question"]))
        print("=" * 20)

        display("=" * 20)
        print(
            f"{self.result_method1}-PREDICT-CONTEXT: {result['retrieval1_is_correct']}"
        )
        display(HTML(result["retrieval1-predict-context"][0]))
        print("-" * 20)
        print(f"{self.result_method1}-VALUE OF {self.result_method1}-PREDICT-CONTEXT: ")
        print(
            f"{self.simplify_data(result['retrieval1-predict-context_retrieval1-values'])}"
        )
        print("-" * 20)
        print(f"{self.result_method2}-VALUE OF {self.result_method1}-PREDICT-CONTEXT: ")
        print(
            self.simplify_data(result["retrieval1-predict-context_retrieval2-values"])
        )
        print("=" * 20)

        display("=" * 20)
        print(
            f"{self.result_method2}-PREDICT-CONTEXT: {result['retrieval2_is_correct']}"
        )
        display(HTML(result["retrieval2-predict-context"][0]))
        print("-" * 20)
        print(f"{self.result_method1}-VALUE OF {self.result_method2}-PREDICT-CONTEXT: ")
        print(
            self.simplify_data(result["retrieval2-predict-context_retrieval1-values"])
        )
        print("-" * 20)
        print(f"{self.result_method2}-VALUE OF {self.result_method2}-PREDICT-CONTEXT: ")
        print(
            self.simplify_data(result["retrieval2-predict-context_retrieval2-values"])
        )
        print("=" * 20)

        display("=" * 20)
        print("ANSWER-CONTEXT: ")
        display(HTML(result["answer-context"]))
        print("-" * 20)
        print(f"{self.result_method1}-VALUE OF ANSWER-CONTEXT: ")
        print(self.simplify_data(result["answer-context_retrieval1-values"]))
        print("-" * 20)
        print(f"{self.result_method2}-VALUE OF ANSWER-CONTEXT: ")
        print(self.simplify_data(result["answer-context_retrieval2-values"]))
        print("=" * 20)
        print()
        print("\n\n\n")

    def steamlit_query_result(self, idx):
        assert self.result_method1 != None

        result = self.result_list[idx]
        st.markdown(f"#### 질문")
        st.write(result["question"])
        st.markdown(f"#### 정답")
        st.write(", ".join(result["answer"]))

        if result["retrieval1_is_correct"]:
            st.markdown(
                f"#### {self.result_method1} 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"#### {self.result_method1} 예측 문서 (<span style='color:red;'>예측 실패</span>)",
                unsafe_allow_html=True,
            )
        st.write(result["retrieval1-predict-context"][0])
        st.markdown("**토큰 측정 점수**")
        df1 = pd.DataFrame(
            self.simplify_data(result["retrieval1-predict-context_retrieval1-values"]),
            columns=["토큰", f"{self.result_method1} 점수"],
        )
        st.dataframe(df1.T)

        st.markdown(f"#### 정답 문서")
        st.write(result["answer-context"])
        st.markdown("**토큰 측정 점수**")
        df2 = pd.DataFrame(
            self.simplify_data(result["answer-context_retrieval1-values"]),
            columns=["토큰", f"{self.result_method1} 점수"],
        )
        st.dataframe(df2.T)

    def streamlit_compare_query_result(self, idx):
        assert self.result_method1 != None and self.result_method2 != None
        result = self.result_list[idx]
        st.markdown(f"#### 질문")
        st.write(result["question"])
        st.markdown(f"#### 정답")
        st.write(", ".join(result["answer"]))

        if result["retrieval1_is_correct"]:
            st.markdown(
                f"#### {self.result_method1} 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"#### {self.result_method1} 예측 문서 (<span style='color:red;'>예측 실패</span>)",
                unsafe_allow_html=True,
            )
        st.write(result["retrieval1-predict-context"][0])
        st.markdown("**토큰 측정 점수**")
        df1 = pd.DataFrame(
            self.simplify_data(result["retrieval1-predict-context_retrieval1-values"]),
            columns=["토큰", f"{self.result_method1} 점수"],
        )
        df2 = pd.DataFrame(
            self.simplify_data(result["retrieval1-predict-context_retrieval2-values"]),
            columns=["토큰", f"{self.result_method2} 점수"],
        )
        st.dataframe(df1.T)
        st.dataframe(df2.T)

        if result["retrieval2_is_correct"]:
            st.markdown(
                f"#### {self.result_method2} 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"#### {self.result_method2} 예측 문서 (<span style='color:red;'>예측 실패</span>)",
                unsafe_allow_html=True,
            )
        st.write(result["retrieval2-predict-context"][0])
        st.markdown("**토큰 측정 점수**")
        df1 = pd.DataFrame(
            self.simplify_data(result["retrieval2-predict-context_retrieval1-values"]),
            columns=["토큰", f"{self.result_method1} 점수"],
        )
        df2 = pd.DataFrame(
            self.simplify_data(result["retrieval2-predict-context_retrieval2-values"]),
            columns=["토큰", f"{self.result_method2} 점수"],
        )
        st.dataframe(df1.T)
        st.dataframe(df2.T)

        st.markdown(f"#### 정답 문서")
        st.write(result["answer-context"])
        st.markdown("**토큰 측정 점수**")
        df1 = pd.DataFrame(
            self.simplify_data(result["answer-context_retrieval1-values"]),
            columns=["토큰", f"{self.result_method1} 점수"],
        )
        df2 = pd.DataFrame(
            self.simplify_data(result["answer-context_retrieval2-values"]),
            columns=["토큰", f"{self.result_method2} 점수"],
        )
        st.dataframe(df1.T)
        st.dataframe(df2.T)
