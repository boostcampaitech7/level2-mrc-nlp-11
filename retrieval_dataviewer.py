import streamlit as st
import pandas as pd

from utils.analysis_sparse import SparseRetrievalResultViewer

st.set_page_config(layout="wide", page_title="SEVEN ELEVEN MRC Data Viewer V1.0.0")


def main():
    tab1, tab2 = st.tabs(["문서 검색 결과 살펴보기", "문서 검색 결과 비교하기"])
    result_path = "/Users/gj/Documents/study/level2-mrc-nlp-11/data/SubwordBm25Retrieval-MorphsBm25Retrieval-compare-result.json"
    result_viewer = SparseRetrievalResultViewer(result_path)

    result_method1 = result_viewer.result_method1
    result_method2 = result_viewer.result_method2

    with tab1:
        setting_section, result_section = st.columns([1, 2])
        with setting_section:
            st.header("질문 선택")

            # 선택 방식을 라디오 버튼으로 변경
            selection_method = st.radio(
                "선택 방식",
                [
                    "직접 입력",
                    f"{result_method1} 예측 성공",
                    f"{result_method1} 예측 실패",
                ],
            )

            final_query_index = None

            if selection_method == "직접 입력":
                query_index_input = st.text_input(
                    "질문 인덱스 직접 입력:", key="input1"
                )
                if query_index_input:
                    final_query_index = query_index_input

            elif selection_method == f"{result_method1} 예측 성공":
                retrieval1_correct_query_idx = (
                    result_viewer.get_retrieval1_correct_query_idx()
                )
                final_query_index = st.selectbox(
                    f"{result_method1} 예측 성공한 질문 선택",
                    retrieval1_correct_query_idx,
                    key="retrieval1_correct_selectbox",
                )

            elif selection_method == f"{result_method1} 예측 실패":
                retrieval1_incorrect_query_idx = (
                    result_viewer.get_retrieval1_incorrect_query_idx()
                )
                final_query_index = st.selectbox(
                    f"{result_method1} 예측 실패한 질문 선택",
                    retrieval1_incorrect_query_idx,
                    key="retrieval1_incorrect_selectbox",
                )

            # 선택된 인덱스 표시
            if final_query_index:
                st.write(f"선택된 질문 인덱스: {final_query_index}")
            else:
                st.write("질문을 선택해주세요.")

        with result_section:
            if final_query_index:
                try:
                    result_viewer.steamlit_query_result(int(final_query_index))
                except (ValueError, IndexError):
                    st.error("유효하지 않은 인덱스입니다. 올바른 숫자를 입력해주세요.")
            else:
                st.info("질문을 선택해주세요.")

    with tab2:
        setting_section, result_section = st.columns([1, 2])
        with setting_section:
            st.header("질문 선택")

            # 선택 방식을 라디오 버튼으로 변경
            selection_method = st.radio(
                "선택 방식",
                [
                    "직접 입력",
                    f"{result_method1}만 예측 성공",
                    f"{result_method2}만 예측 성공",
                    "둘다 예측 실패",
                ],
            )

            final_query_index = None

            if selection_method == "직접 입력":
                query_index_input = st.text_input(
                    "질문 인덱스 직접 입력:", key="input2"
                )
                if query_index_input:
                    final_query_index = query_index_input

            elif selection_method == f"{result_method1}만 예측 성공":
                only_retrieval1_correct_query_idx = (
                    result_viewer.get_only_retrieval1_correct_query_idx()
                )
                final_query_index = st.selectbox(
                    f"{result_method1}만 예측 성공한 질문 선택",
                    only_retrieval1_correct_query_idx,
                    key="retrieval1_selectbox",
                )

            elif selection_method == f"{result_method2}만 예측 성공":
                only_retrieval2_correct_query_idx = (
                    result_viewer.get_only_retrieval2_correct_query_idx()
                )
                final_query_index = st.selectbox(
                    f"{result_method2}만 예측 성공한 질문 선택",
                    only_retrieval2_correct_query_idx,
                    key="retrieval2_selectbox",
                )

            elif selection_method == "둘다 예측 실패":
                both_incorrect_query_idx = result_viewer.get_both_incorrect_query_idx()
                final_query_index = st.selectbox(
                    "둘다 예측 실패한 질문 선택",
                    both_incorrect_query_idx,
                    key="retrieval2_selectbox",
                )

            # 선택된 인덱스 표시
            if final_query_index:
                st.write(f"선택된 질문 인덱스: {final_query_index}")
            else:
                st.write("질문을 선택해주세요.")

        with result_section:
            if final_query_index:
                try:
                    result_viewer.streamlit_compare_query_result(int(final_query_index))
                except (ValueError, IndexError):
                    st.error("유효하지 않은 인덱스입니다. 올바른 숫자를 입력해주세요.")
            else:
                st.info("질문을 선택해주세요.")


if __name__ == "__main__":
    main()
