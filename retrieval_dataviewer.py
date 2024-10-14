import streamlit as st
import hydra
import pandas as pd
from utils.analysis_sparse import SparseRetrievalAnalysis
import os


st.set_page_config(layout="wide", page_title="SEVEN ELEVEN MRC Data Viewer V1.0.0")


def view_result(result, analyzer):
    st.markdown(f"#### 질문")
    st.write(result["question"])
    st.markdown(f"#### 정답")
    st.write(", ".join(result["answer"]))

    if result["retrieval1_is_correct"]:
        st.markdown(
            f"#### RETRIEVAL1 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"#### RETRIEVAL1 예측 문서 (<span style='color:red;'>예측 실패</span>)",
            unsafe_allow_html=True,
        )
    st.write(result["retrieval1-predict-context"][0])
    st.markdown("**토큰 측정 점수**")
    df = pd.DataFrame(
        analyzer.simplify_data(result["retrieval1-predict-context_retrieval1-values"]),
        columns=["토큰", "RETRIEVAL1 점수"],
    )
    st.dataframe(df.T)

    st.markdown(f"#### 정답 문서")
    st.write(result["answer-context"])
    st.markdown("**토큰 측정 점수**")
    df = pd.DataFrame(
        analyzer.simplify_data(result["answer-context_retrieval1-values"]),
        columns=["토큰", "RETRIEVAL1 점수"],
    )
    st.dataframe(df.T)


def view_compare_result(result, analyzer):
    st.markdown(f"#### 질문")
    st.write(result["question"])
    st.markdown(f"#### 정답")
    st.write(", ".join(result["answer"]))

    if result["retrieval1_is_correct"]:
        st.markdown(
            f"#### RETRIEVAL1 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"#### RETRIEVAL1 예측 문서 (<span style='color:red;'>예측 실패</span>)",
            unsafe_allow_html=True,
        )
    st.write(result["retrieval1-predict-context"][0])
    st.markdown("**토큰 측정 점수**")
    df1 = pd.DataFrame(
        analyzer.simplify_data(result["retrieval1-predict-context_retrieval1-values"]),
        columns=["토큰", "RETRIEVAL1 점수"],
    )
    df2 = pd.DataFrame(
        analyzer.simplify_data(result["retrieval1-predict-context_retrieval2-values"]),
        columns=["토큰", "RETRIEVAL2 점수"],
    )
    df_merged = pd.merge(df1, df2, on="토큰")
    df_transposed = df_merged.T
    st.dataframe(df_transposed)

    if result["retrieval2_is_correct"]:
        st.markdown(
            f"#### RETRIEVAL2 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"#### RETRIEVAL2 예측 문서 (<span style='color:red;'>예측 실패</span>)",
            unsafe_allow_html=True,
        )
    st.write(result["retrieval2-predict-context"][0])
    st.markdown("**토큰 측정 점수**")
    df1 = pd.DataFrame(
        analyzer.simplify_data(result["retrieval2-predict-context_retrieval1-values"]),
        columns=["토큰", "RETRIEVAL1 점수"],
    )
    df2 = pd.DataFrame(
        analyzer.simplify_data(result["retrieval2-predict-context_retrieval2-values"]),
        columns=["토큰", "RETRIEVAL2 점수"],
    )
    df_merged = pd.merge(df1, df2, on="토큰")
    df_transposed = df_merged.T
    st.dataframe(df_transposed)

    st.markdown(f"#### 정답 문서")
    st.write(result["answer-context"])
    st.markdown("**토큰 측정 점수**")
    df1 = pd.DataFrame(
        analyzer.simplify_data(result["answer-context_retrieval1-values"]),
        columns=["토큰", "RETRIEVAL1 점수"],
    )
    df2 = pd.DataFrame(
        analyzer.simplify_data(result["answer-context_retrieval2-values"]),
        columns=["토큰", "RETRIEVAL2 점수"],
    )
    df_merged = pd.merge(df1, df2, on="토큰")
    df_transposed = df_merged.T
    st.dataframe(df_transposed)


@hydra.main(config_path="./config", config_name="retrieval", version_base=None)
def main(config):
    tab1, tab2 = st.tabs(["문서 검색 결과 살펴보기", "문서 검색 결과 비교하기"])
    anaylzer = SparseRetrievalAnalysis()

    with tab1:
        anaylzer.load_result(
            "/data/ephemeral/home/gj/level2-mrc-nlp-11/data/TfIdfRetrieval-result.json"
        )
        setting_section, result_section = st.columns([1, 2])
        with setting_section:
            st.header("질문 선택")

            # 선택 방식을 라디오 버튼으로 변경
            selection_method = st.radio(
                "선택 방식",
                ["직접 입력", "RETRIEVAL1 예측 성공", "RETRIEVAL1 예측 실패"],
            )

            final_query_index = None

            if selection_method == "직접 입력":
                query_index_input = st.text_input("질문 인덱스 직접 입력:")
                if query_index_input:
                    final_query_index = query_index_input

            elif selection_method == "RETRIEVAL1 예측 성공":
                retrieval1_correct_query_idx = (
                    anaylzer.get_retrieval1_correct_query_idx()
                )
                final_query_index = st.selectbox(
                    "RETRIEVAL1 예측 성공한 질문 선택",
                    retrieval1_correct_query_idx,
                    key="retrieval1_correct_selectbox",
                )

            elif selection_method == "RETRIEVAL1 예측 실패":
                retrieval1_incorrect_query_idx = (
                    anaylzer.get_retrieval1_incorrect_query_idx()
                )
                final_query_index = st.selectbox(
                    "RETRIEVAL2 예측 실패한 질문 선택",
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
                    view_result(anaylzer.result_list[int(final_query_index)], anaylzer)
                except (ValueError, IndexError):
                    st.error("유효하지 않은 인덱스입니다. 올바른 숫자를 입력해주세요.")
            else:
                st.info("질문을 선택해주세요.")

    with tab2:
        anaylzer.load_result(
            "/data/ephemeral/home/gj/level2-mrc-nlp-11/data/TfIdfRetrieval-SubwordBm25Retrieval-compare-result.json"
        )
        setting_section, result_section = st.columns([1, 2])
        with setting_section:
            st.header("질문 선택")

            # 선택 방식을 라디오 버튼으로 변경
            selection_method = st.radio(
                "선택 방식",
                [
                    "직접 입력",
                    "RETRIEVAL1 예측 성공",
                    "RETRIEVAL2 예측 성공",
                    "둘다 예측 실패",
                ],
            )

            final_query_index = None

            if selection_method == "직접 입력":
                query_index_input = st.text_input("질문 인덱스 직접 입력:")
                if query_index_input:
                    final_query_index = query_index_input

            elif selection_method == "RETRIEVAL1 예측 성공":
                only_retrieval1_correct_query_idx = (
                    anaylzer.get_only_retrieval1_correct_query_idx()
                )
                final_query_index = st.selectbox(
                    "RETRIEVAL1 예측 성공한 질문 선택",
                    only_retrieval1_correct_query_idx,
                    key="retrieval1_selectbox",
                )

            elif selection_method == "RETRIEVAL2 예측 성공":
                only_retrieval2_correct_query_idx = (
                    anaylzer.get_only_retrieval2_correct_query_idx()
                )
                final_query_index = st.selectbox(
                    "RETRIEVAL2 예측 성공한 질문 선택",
                    only_retrieval2_correct_query_idx,
                    key="retrieval2_selectbox",
                )

            elif selection_method == "둘다 예측 실패":
                both_incorrect_query_idx = anaylzer.get_both_incorrect_query_idx()
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
                    view_compare_result(
                        anaylzer.result_list[int(final_query_index)], anaylzer
                    )
                except (ValueError, IndexError):
                    st.error("유효하지 않은 인덱스입니다. 올바른 숫자를 입력해주세요.")
            else:
                st.info("질문을 선택해주세요.")


if __name__ == "__main__":
    main()
