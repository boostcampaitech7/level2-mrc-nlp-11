import streamlit as st
from dotenv import load_dotenv
import os, json
from module.data import MrcDataModule
from utils.dataviewer_components import view_answer, view_predictions
from datasets import load_from_disk

load_dotenv()


@st.cache_data
def load_predictions(prediction_path):
    with open(prediction_path, "r", encoding="utf-8") as f:
        nbest_predictions = json.load(f)
    return nbest_predictions


def view_train_data(data_module, config):
    train_examples = data_module.train_examples
    select_options = [
        f'{example["id"]}: {example["question"]}' for example in train_examples
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)

    selected_example = train_examples[select_options.index(selected_question)]

    answer_column, prediction_column = st.columns(2, gap="medium")

    with answer_column:
        view_answer(selected_example)

    with prediction_column:
        view_predictions(selected_example, None)


def view_validation_data(data_module, config):
    nbest_predictions = load_predictions(
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11" + config.prediction.validation
    )
    validation_examples = data_module.eval_examples
    select_options = [
        f'{example["id"]}: {example["question"]}' for example in validation_examples
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)

    selected_example = validation_examples[select_options.index(selected_question)]

    answer_column, prediction_column = st.columns(2, gap="medium")

    with answer_column:
        view_answer(selected_example)

    with prediction_column:
        selected_nbest_prediction = nbest_predictions[selected_example["id"]]
        view_predictions(selected_example, selected_nbest_prediction)


def view_test_data(data_module, config):
    nbest_predictions = load_predictions(
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11" + config.prediction.test
    )
    test_examples = load_from_disk(
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/test_dataset"
    )["validation"]

    select_options = [
        f'{example["id"]}: {example["question"]}' for example in test_examples
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)

    selected_example = test_examples[select_options.index(selected_question)]

    answer_column, prediction_column = st.columns(2, gap="medium")

    with answer_column:
        view_answer(selected_example)

    with prediction_column:
        selected_nbest_prediction = nbest_predictions[selected_example["id"]]
        view_predictions(selected_example, selected_nbest_prediction)
