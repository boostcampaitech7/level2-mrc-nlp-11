import streamlit as st
import numpy as np
from datasets import load_from_disk
import json
import hydra
from module.data import *
import pandas as pd


# 화면 레이아웃 설정
st.set_page_config(layout="wide", page_title="SEVEN ELEVEN MRC Data Viewer V1.0.0")


@st.cache_data
def load_data(_config):
    """
    질문-문서(-정답) 페어 데이터셋을 Dataframe 형식으로 불러오는 함수입니다.
    Args:
        _config: 템플릿의 config/mrc.yaml 파일 기준으로 작성되어 있습니다. 템플릿을 사용하지 않을 경우 지우거나 주석처리하고 사용하려는 데이터 경로를 지정해주세요.
    Returns:
        dataset: 질문-문서(-정답) 페어 데이터셋입니다.
    """

    # data_path = (
    #     os.path.dirname(os.path.abspath(__file__)) + f"/data/train_dataset/train"
    # )  # 베이스라인 데이터의 train_dataset/train
    data_path = (
        os.path.dirname(os.path.abspath(__file__)) + f"/data/train_dataset/validation"
    )  # 베이스라인 데이터의 train_dataset/validation

    # # config에 설정한 데이터셋 불러오기
    # # **미구현**
    # dataset_list = get_dataset_list(_config.data.dataset_name)

    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
        dataset = dataset.to_pandas()
        return dataset
    else:
        print("⚠️지정한 경로에 데이터셋 파일이 존재하지 않습니다.")
        return None


@st.cache_data
def load_predictions(_config):
    """
    질문에 대한 nbest 예측 데이터를 Object 형식으로 불러오는 함수입니다.
    Args:
        _config: 템플릿의 config/mrc.yaml 파일 기준으로 작성되어 있습니다. 템플릿을 사용하지 않을 경우 지우거나 주석처리하고 사용하려는 데이터 경로를 지정해주세요.
    Returns:
        predictions: 질문에 대한 nbest 예측 데이터
    """
    # config에 설정한 output dir 경로의 nbest prediction
    predictions_path = f"{_config.train.output_dir}/eval_nbest_predictions.json"
    # prediction_path = os.path.dirname(os.path.abspath(__file__)) + "/outputs/eval_nbest_predictions.json"

    if os.path.exists(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        return predictions
    else:
        print("⚠️지정한 경로에 nbest prediction 파일이 존재하지 않습니다.")
        return None


@st.cache_data
def load_tokenized_samples(_config):
    """
    질문-문서 시퀀스의 토크나이징 결과 데이터를 Object 형식으로 불러오는 함수입니다.
    토큰 데이터가 없는 경우 원하는 데이터에 대해 utils/save_tokenized_samples.py를 실행하세요.
    Args:
        _config: 템플릿의 config/mrc.yaml 파일 기준으로 작성되어 있습니다. 템플릿을 사용하지 않을 경우 지우거나 주석처리하고 사용하려는 데이터 경로를 지정해주세요.
    Returns:
        tokenized_samples: 질문-문서 시퀀스의 토크나이징 결과 데이터
    """
    tokenized_samples_path = f"{_config.train.output_dir}/eval_tokenized_samples.json"
    # tokenized_samples_path = os.path.dirname(os.path.abspath(__file__)) + "/outputs/tokenized_samples.json"

    if os.path.exists(tokenized_samples_path):
        with open(tokenized_samples_path, "r", encoding="utf-8") as f:
            tokenized_samples = json.load(f)
        return tokenized_samples
    else:
        print("⚠️지정한 경로에 tokenized sample 파일이 존재하지 않습니다.")
        return None


@st.cache_data
def load_wiki():
    wiki_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/data/wikipedia_documents.json"
    )
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki_data = json.load(f)
        # text 값 기준으로 중복 제거하여 반환
        wiki_unique = {}
        for v in wiki_data.values():
            if v["text"] not in wiki_unique.keys():
                wiki_unique[v["text"]] = v
        return list(wiki_unique.values())
    else:
        print("⚠️지정한 경로에 wiki documents 파일이 존재하지 않습니다.")
        return None


@st.cache_data
def load_tfidf_info(_config):
    tfidf_info_path = f"{_config.train.output_dir}/wiki_tfidf_info.json"

    if os.path.exists(tfidf_info_path):
        with open(tfidf_info_path, "r", encoding="utf-8") as f:
            tfidf_info = json.load(f)
        return tfidf_info
    else:
        print("⚠️지정한 경로에 tfidf info 파일이 존재하지 않습니다.")
        return None


@st.cache_data
def load_retrieved_documents(_config):
    test_dataset_path = (
        os.path.dirname(os.path.abspath(__file__))
        + f"/data/default/test_dataset/validation"
    )
    retrieved_documents_path = f"{_config.train.output_dir}/retrieved_documents.json"

    if os.path.exists(test_dataset_path):
        test_dataset = load_from_disk(test_dataset_path)
        test_dataset = test_dataset.to_pandas()
    else:
        print("⚠️지정한 경로에 데이터셋 파일이 존재하지 않습니다.")
        test_dataset = None

    if os.path.exists(retrieved_documents_path):
        with open(retrieved_documents_path, "r", encoding="utf-8") as f:
            retrieved_documents = json.load(f)
    else:
        print("⚠️지정한 경로에 retrieved documents 파일이 존재하지 않습니다.")
        retrieved_documents = None

    return test_dataset, retrieved_documents


def view_QA(question_id, data, prediction):
    """
    선택한 질문-문서 페어를 보기 좋게 화면에 표시하는 함수입니다.
    Args:
        question_id: 선택한 질문 id
        data: question_id에 대응하는 질문-문서 페어 데이터
        prediction: question_id에 대응하는 예측 데이터(없으면 표시 안 됨)
    Returns:
        None
    """
    if not data.empty:
        # 질문, 답변, 그리고 문서 내용을 표시
        st.write(f"**질문 ID**: {question_id}")
        st.write(f"**질문 내용**: {data['question']}")

        document_title = data["title"]
        document_content = data["context"]
        answer_text = data["answers"]["text"][0]
        answer_start = data["answers"]["answer_start"][0]
        answer_end = answer_start + len(answer_text)

        st.write(f"**문서 ID**: {data['document_id']}")
        st.write(f"**문서 제목**: {document_title}")

        # 문서 내용, 정답/예측에 하이라이트 표시
        if question_id in prediction.keys():
            prediction_text = prediction[question_id][0]["text"]
            prediction_start = prediction[question_id][0]["start"]
            prediction_end = prediction_start + len(prediction_text)

            answer_prediction_span = ""
            for i in range(
                min(prediction_start, answer_start), max(prediction_end, answer_end)
            ):
                char = document_content[i]
                if answer_start <= i < answer_end:
                    char = f"<span style='background-color: #FD7E84;'>{char}</span>"
                if prediction_start <= i < prediction_end:
                    char = (
                        f"<span style='border-bottom: 3px solid #2E9AFE;'>{char}</span>"
                    )
                answer_prediction_span += char

            highlighted_content = (
                document_content[: min(answer_start, prediction_start)]
                + answer_prediction_span
                + document_content[max(answer_end, prediction_end) :]
            )
        else:
            highlighted_content = (
                document_content[:answer_start]
                + f"<span style='background-color: #FD7E84;'>{answer_text}</span>"
                + document_content[answer_start + len(answer_text) :]
            )
        st.markdown(highlighted_content, unsafe_allow_html=True)

        # 질문에 대한 정답/예측 표시
        st.markdown(
            f"**정답 텍스트**: <span style='background-color: #FD7E84;'>{answer_text}</span> (start: {answer_start})",
            unsafe_allow_html=True,
        )
        if question_id in prediction.keys():
            st.markdown(
                f"**예측 텍스트**: <span style='border-bottom: 3px solid #2E9AFE;'>{prediction[question_id][0]['text']}</span> (start: {prediction[question_id][0]['start']})",
                unsafe_allow_html=True,
            )
            with st.expander("예측 nbest 확인"):
                for idx, pred in enumerate(prediction[question_id]):
                    st.write(f"{idx+1}: {pred['text']} (start: {pred['start']})")
    else:
        st.write("선택된 문서와 질문 ID에 해당하는 데이터가 없습니다.")


@hydra.main(config_path="./config", config_name="mrc", version_base=None)
def main(config):
    """
    전체 화면 요소를 그리는 함수입니다. (추후 모듈화, 재사용성 보완 예정.)
    """
    # load data
    dataset = load_data(config)
    predictions = load_predictions(config)
    tokenized_samples = load_tokenized_samples(config)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["하나씩 보기", "묶음으로 보기", "위키 문서 살펴보기", "Retrieval 결과 보기"]
    )
    # ================
    # 하나씩 보기 탭
    # ================
    with tab1:
        setting_section, raw_data_section, analysis_section = st.columns([1, 2, 3])

        # ===============
        # setting_section
        # ===============
        with setting_section:
            st.header("필터링 옵션")
            document_options = ["0: 전체 보기"]
            for _, row in dataset.iterrows():
                document_options.append(
                    ": ".join([str(row["document_id"]), row["title"]])
                )
            selected_document = st.selectbox(
                "문서 선택 (0 = 전체 보기)", document_options
            )
            selected_document_id = int(selected_document.split(":")[0])

            question_list = (
                dataset[dataset["document_id"] == selected_document_id]["id"]
                + ": "
                + dataset[dataset["document_id"] == selected_document_id]["question"]
                if selected_document_id
                else dataset["id"] + ": " + dataset["question"]
            )
            selected_question = st.selectbox("질문 선택", question_list)
            selected_question_id = selected_question.split(":")[0]

            # 선택한 문서와 질문에 해당하는 데이터 필터링
            filtered_data = dataset[dataset["id"] == selected_question_id]

        # ================
        # raw_data_section
        # ================
        with raw_data_section:
            view_QA(selected_question_id, filtered_data.iloc[0], predictions)

        # ================
        # analysis_section
        # ================
        with analysis_section:
            if not filtered_data.empty:
                st.subheader("토큰화 결과")

                if selected_question_id in tokenized_samples.keys():
                    colored_tokens = []
                    for token in tokenized_samples[selected_question_id]:
                        # 스페셜 토큰 처리
                        if token in ["[SEP]", "[CLS]"]:
                            colored_tokens.append(
                                f"<div style='display:inline-block; font-size:14px; background-color:#c5cff6; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                            )
                        # 패드 토큰
                        elif token == "[PAD]":
                            colored_tokens.append(
                                f"<div style='display:inline-block; font-size:14px; color: #ddd; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                            )
                        else:
                            colored_tokens.append(
                                f"<div style='display:inline-block; font-size:14px; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                            )

                    st.markdown(" ".join(colored_tokens), unsafe_allow_html=True)
                else:
                    st.write("해당 쿼리에 대한 토큰화 정보가 없습니다🫠")

    # ================
    # 묶음으로 보기 탭
    # ================
    with tab2:
        # 한 페이지에 표시할 항목 개수 설정
        items_per_page = st.selectbox(
            "페이지 당 표시할 질문 수", [5, 10, 15, 20], index=0
        )

        # 전체 페이지 수 계산
        total_pages = len(dataset) // items_per_page + (
            1 if len(dataset) % items_per_page > 0 else 0
        )

        # 현재 페이지 선택 슬라이더
        current_page = st.slider("페이지 선택", 1, total_pages, 1)

        # 현재 페이지에 해당하는 데이터만 선택
        start_idx = (current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = dataset.iloc[start_idx:end_idx]

        # 현재 페이지의 데이터 출력
        st.write(f"### 현재 페이지: {current_page} / {total_pages}")

        for idx, row in page_data.iterrows():
            # 각 질문-답변 페어 표시
            q_id = row["id"]
            with st.expander(f"{q_id} : {row['question']}"):
                col1, col2 = st.columns([2, 3])
                with col1:
                    if not row.empty:
                        # 질문, 답변, 그리고 문서 내용을 표시
                        st.write(f"**질문 ID**: {q_id}")
                        st.write(f"**질문 내용**: {row['question']}")

                        document_title = row["title"]
                        document_content = row["context"]
                        answer_text = row["answers"]["text"][0]
                        answer_start = row["answers"]["answer_start"][0]
                        answer_end = answer_start + len(answer_text)

                        st.write(f"**문서 ID**: {row['document_id']}")
                        st.write(f"**문서 제목**: {document_title}")

                        # 문서 내용, 정답/예측에 하이라이트 표시
                        if q_id in predictions.keys():
                            prediction_text = predictions[q_id][0]["text"]
                            prediction_start = predictions[q_id][0]["start"]
                            prediction_end = prediction_start + len(prediction_text)

                            answer_prediction_span = ""
                            for i in range(
                                min(prediction_start, answer_start),
                                max(prediction_end, answer_end),
                            ):
                                char = document_content[i]
                                if answer_start <= i < answer_end:
                                    char = f"<span style='background-color: #FD7E84;'>{char}</span>"
                                if prediction_start <= i < prediction_end:
                                    char = f"<span style='border-bottom: 3px solid #2E9AFE;'>{char}</span>"
                                answer_prediction_span += char

                            highlighted_content = (
                                document_content[: min(answer_start, prediction_start)]
                                + answer_prediction_span
                                + document_content[max(answer_end, prediction_end) :]
                            )
                        else:
                            highlighted_content = (
                                document_content[:answer_start]
                                + f"<span style='background-color: #FD7E84;'>{answer_text}</span>"
                                + document_content[answer_start + len(answer_text) :]
                            )
                        st.markdown(highlighted_content, unsafe_allow_html=True)

                        # 질문에 대한 정답/예측 표시
                        st.markdown(
                            f"**정답 텍스트**: <span style='background-color: #FD7E84;'>{answer_text}</span> (start: {answer_start})",
                            unsafe_allow_html=True,
                        )
                        if q_id in predictions.keys():
                            st.markdown(
                                f"**예측 텍스트**: <span style='border-bottom: 3px solid #2E9AFE;'>{predictions[q_id][0]['text']}</span> (start: {predictions[q_id][0]['start']})",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.write("선택된 문서와 질문 ID에 해당하는 데이터가 없습니다.")

                with col2:
                    if not row.empty:
                        st.subheader("토큰화 결과")

                        if q_id in tokenized_samples.keys():
                            colored_tokens = []
                            for token in tokenized_samples[q_id]:
                                # 스페셜 토큰 처리
                                if token in ["[SEP]", "[CLS]"]:
                                    colored_tokens.append(
                                        f"<div style='display:inline-block; font-size:14px; background-color:#c5cff6; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                                    )
                                # 패드 토큰
                                elif token == "[PAD]":
                                    colored_tokens.append(
                                        f"<div style='display:inline-block; font-size:14px; color: #ddd; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                                    )
                                else:
                                    colored_tokens.append(
                                        f"<div style='display:inline-block; font-size:14px; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                                    )

                            st.markdown(
                                " ".join(colored_tokens), unsafe_allow_html=True
                            )
                        else:
                            st.write("해당 쿼리에 대한 토큰화 정보가 없습니다🫠")

    # ================
    # 위키 문서 살펴보기 탭
    # ================
    with tab3:
        wiki_list = load_wiki()
        tfidf_infos = load_tfidf_info(config)

        # 한 페이지에 표시할 항목 개수 설정
        documents_per_page = st.selectbox(
            "페이지 당 표시할 문서 수", [10, 20, 30], index=0
        )

        # 전체 페이지 수 계산
        total_document_pages = len(wiki_list) // documents_per_page + (
            1 if len(wiki_list) % documents_per_page > 0 else 0
        )

        # 현재 페이지 선택 슬라이더
        document_page = st.slider(
            "페이지 선택", 1, total_document_pages, 1, key="document_slider"
        )

        # 현재 페이지에 해당하는 데이터만 선택
        start_idx = (document_page - 1) * documents_per_page
        end_idx = start_idx + documents_per_page
        selected_documents = wiki_list[start_idx:end_idx]

        # 현재 페이지의 데이터 출력
        st.write(f"### 현재 페이지: {document_page} / {total_document_pages}")

        for idx, wiki_document in enumerate(selected_documents):
            tfidf_info = tfidf_infos[str(start_idx + idx)]
            with st.expander(
                f"{wiki_document['document_id']}: {wiki_document['title']}"
            ):
                st.write(
                    f"Domain: {wiki_document['domain']} | Corpus Source: {wiki_document['corpus_source']} | Author: {wiki_document['author']} html: {wiki_document['html']} | url: {wiki_document['url']}"
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Markdown")
                    st.write(wiki_document["text"])
                with c2:
                    st.subheader("Raw text")
                    st.text(wiki_document["text"])

                st.subheader("TF-IDF 점수 상위 10개 토큰")
                tfidf_text = []
                for t, score in tfidf_info.items():
                    tfidf_text.append(
                        f"<div style='display:inline-block; font-size:14px; background-color:#c5cff6; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{t}</div>: {round(score, 3)}"
                    )
                st.markdown(", ".join(tfidf_text), unsafe_allow_html=True)

    with tab4:
        test_dataset, retrieved_documents = load_retrieved_documents(config)


if __name__ == "__main__":
    main()
