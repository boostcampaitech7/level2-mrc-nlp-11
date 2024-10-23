import streamlit as st
from dotenv import load_dotenv
import os, json
from module.data import MrcDataModule
from utils.dataviewer_components import view_answer, view_predictions, view_documents
from datasets import load_from_disk

load_dotenv()


@st.cache_data
def load_examples(mode):
    if mode == "train":
        examples = load_from_disk(
            os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/default/train"
        )
    elif mode == "validation":
        examples = load_from_disk(
            os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/default/validation"
        )
    else:
        examples = load_from_disk(
            os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/test_dataset/validation"
        )

    return examples


@st.cache_data
def load_predictions(prediction_path):
    with open(prediction_path, "r", encoding="utf-8") as f:
        nbest_predictions = json.load(f)
    return nbest_predictions


@st.cache_data
def load_wiki():
    wiki_path = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/data/normalized_wikipedia_documents.json"
    )
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    wiki_list = [doc for doc in wiki.values()]
    return wiki_list


def view_train_data(data_module, config):
    train_examples = load_examples("train")
    select_options = [
        f'{example["id"]}: {example["question"]}' for example in train_examples
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)

    selected_example = train_examples[select_options.index(selected_question)]

    answer_column, prediction_column = st.columns(2, gap="medium")

    with answer_column:
        view_answer(data_module, selected_example)

    with prediction_column:
        view_predictions(data_module, selected_example, None)


def view_validation_data(data_module, config):
    nbest_predictions = load_predictions(
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11" + config.prediction.validation
    )
    validation_examples = load_examples("validation")
    select_options = [
        f'{example["id"]}: {example["question"]}' for example in validation_examples
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)

    selected_example = validation_examples[select_options.index(selected_question)]

    answer_column, prediction_column = st.columns(2, gap="medium")

    with answer_column:
        view_answer(data_module, selected_example)

    with prediction_column:
        selected_nbest_prediction = nbest_predictions[selected_example["id"]]
        view_predictions(data_module, selected_example, selected_nbest_prediction)


def view_test_data(data_module, config):
    nbest_predictions = load_predictions(
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11" + config.prediction.test
    )
    test_examples = load_examples("test")

    select_options = [
        f'{example["id"]}: {example["question"]}' for example in test_examples
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)

    selected_example = test_examples[select_options.index(selected_question)]

    answer_column, prediction_column = st.columns(2, gap="medium")

    with answer_column:
        view_answer(data_module, selected_example)

    with prediction_column:
        selected_nbest_prediction = nbest_predictions[selected_example["id"]]
        view_predictions(data_module, selected_example, selected_nbest_prediction)


def view_wiki():
    wiki = load_wiki()

    if "wikipage" not in st.session_state:
        st.session_state.wikipage = 1

    total_num_wiki = len(wiki)

    items_per_page = 10
    total_pages = total_num_wiki // items_per_page

    # 페이지 전환 버튼
    prev_ten_button, prev_button, page_input, next_button, next_ten_button = st.columns(
        [1, 1, 4, 1, 1], vertical_alignment="bottom"
    )
    with prev_ten_button:
        if (
            st.button("Previous 10", use_container_width=True)
            and st.session_state.wikipage > 1
        ):
            st.session_state.wikipage = max(st.session_state.wikipage - 10, 0)
    with prev_button:
        if (
            st.button("Previous", use_container_width=True)
            and st.session_state.wikipage > 1
        ):
            st.session_state.wikipage -= 1
    with next_button:
        if (
            st.button("Next", use_container_width=True)
            and st.session_state.wikipage < total_pages
        ):
            st.session_state.wikipage += 1
    with next_ten_button:
        if (
            st.button("Next 10", use_container_width=True)
            and st.session_state.wikipage < total_pages
        ):
            st.session_state.wikipage = min(st.session_state.wikipage + 10, total_pages)

    # 페이지 숫자 입력
    with page_input:
        page_in = st.number_input(
            "페이지 입력",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.wikipage,
            step=1,
        )
        if page_in != st.session_state.wikipage:
            st.session_state.wikipage = page_in

    view_documents(
        wiki[
            st.session_state.wikipage
            * items_per_page : (st.session_state.wikipage + 1)
            * items_per_page
        ]
    )
