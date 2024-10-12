import streamlit as st
import json
import hydra
import pandas as pd

from utils.analysis_sparse import SparseRetrievalAnalysis

st.set_page_config(layout="wide", page_title="SEVEN ELEVEN MRC Data Viewer V1.0.0")


def view_result(result, analyzer):
    st.markdown(f"#### 질문")
    st.write(result["question"])

    if result["tfidf_is_correct"]:
        st.markdown(
            f"#### TF-IDF 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"#### TF-IDF 예측 문서 (<span style='color:red;'>예측 실패</span>)",
            unsafe_allow_html=True,
        )
    st.write(result["tfidf-predict-context"][0])
    st.markdown("**토큰 측정 점수**")
    df1 = pd.DataFrame(
        analyzer.simplify_data(result["tfidf-predict-context_tfidf-values"]),
        columns=["토큰", "TF-IDF 점수"],
    )
    df2 = pd.DataFrame(
        analyzer.simplify_data(result["tfidf-predict-context_bm25-values"]),
        columns=["토큰", "BM25 점수"],
    )
    df_merged = pd.merge(df1, df2, on="토큰")
    df_transposed = df_merged.T
    st.dataframe(df_transposed)

    if result["bm25_is_correct"]:
        st.markdown(
            f"#### BM25 예측 문서 (<span style='color:blue;'>예측 성공</span>)",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"#### BM25 예측 문서 (<span style='color:red;'>예측 실패</span>)",
            unsafe_allow_html=True,
        )
    st.write(result["bm25-predict-context"][0])
    st.markdown("**토큰 측정 점수**")
    df1 = pd.DataFrame(
        analyzer.simplify_data(result["bm25-predict-context_tfidf-values"]),
        columns=["토큰", "TF-IDF 점수"],
    )
    df2 = pd.DataFrame(
        analyzer.simplify_data(result["bm25-predict-context_bm25-values"]),
        columns=["토큰", "BM25 점수"],
    )
    df_merged = pd.merge(df1, df2, on="토큰")
    df_transposed = df_merged.T
    st.dataframe(df_transposed)

    st.markdown(f"#### 정답 문서")
    st.write(result["answer-context"])
    st.markdown("**토큰 측정 점수**")
    df1 = pd.DataFrame(
        analyzer.simplify_data(result["answer-context_tfidf-values"]),
        columns=["토큰", "TF-IDF 점수"],
    )
    df2 = pd.DataFrame(
        analyzer.simplify_data(result["answer-context_bm25-values"]),
        columns=["토큰", "BM25 점수"],
    )
    df_merged = pd.merge(df1, df2, on="토큰")
    df_transposed = df_merged.T
    st.dataframe(df_transposed)


@hydra.main(config_path="./config", config_name="retrieval", version_base=None)
def main(config):
    tab1, tab2 = st.tabs(["문서 검색 결과 살펴보기", "ETC"])
    anaylzer = SparseRetrievalAnalysis(config)
    anaylzer.load_result("data/sparse_retrieval_compare_top-1.json")

    with tab1:
        setting_section, result_section = st.columns([1, 2])
        with setting_section:
            st.header("질문 선택")

            # 선택 방식을 라디오 버튼으로 변경
            selection_method = st.radio(
                "선택 방식",
                ["직접 입력", "TF-IDF 예측 성공", "BM25 예측 성공", "둘다 예측 실패"],
            )

            final_query_index = None

            if selection_method == "직접 입력":
                query_index_input = st.text_input("질문 인덱스 직접 입력:")
                if query_index_input:
                    final_query_index = query_index_input

            elif selection_method == "TF-IDF 예측 성공":
                tfidf_correct_query_idx = anaylzer.get_tfidf_correct_query_idx()
                final_query_index = st.selectbox(
                    "TF-IDF 예측 성공한 질문 선택",
                    tfidf_correct_query_idx,
                    key="tfidf_selectbox",
                )

            elif selection_method == "BM25 예측 성공":
                bm25_correct_query_idx = anaylzer.get_bm25_correct_query_idx()
                final_query_index = st.selectbox(
                    "BM25 예측 성공한 질문 선택",
                    bm25_correct_query_idx,
                    key="bm25_selectbox",
                )

            elif selection_method == "둘다 예측 실패":
                both_incorrect_query_idx = anaylzer.get_both_incorrect_query_idx()
                final_query_index = st.selectbox(
                    "둘다 예측 실패한 질문 선택",
                    both_incorrect_query_idx,
                    key="bm25_selectbox",
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


if __name__ == "__main__":
    main()
