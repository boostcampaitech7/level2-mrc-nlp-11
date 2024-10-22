import streamlit as st
import numpy as np
from datasets import load_from_disk
import json
import hydra
from module.data import *
import os
from dotenv import load_dotenv
from utils.dataviewer_tabs import view_train_data, view_validation_data, view_test_data


load_dotenv()


@st.cache_data
def load_wiki():
    wiki_path = (
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/wikipedia_documents.json"
    )
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    return wiki


@hydra.main(config_path="./config", config_name="streamlit_mrc", version_base=None)
def main(config):
    # 화면 레이아웃 설정
    st.set_page_config(layout="wide", page_title="SEVEN ELEVEN ODQA Data Viewer V2.0.0")

    # 우선 default 데이터셋만 볼 수 있게 함
    mrc_config = config.mrc
    streamlit_config = config.streamlit
    mrc_config.data.dataset_name = ["default"]
    data_module = MrcDataModule(mrc_config)
    data_module.setup()

    train_tab, validation_tab, test_tab, wiki_tab = st.tabs(
        ["Train", "Validation", "Test", "Wiki"]
    )

    with train_tab:
        view_train_data(data_module, streamlit_config)

    with validation_tab:
        view_validation_data(data_module, streamlit_config)

    with test_tab:
        view_test_data(data_module, streamlit_config)

    with wiki_tab:
        # view_wiki_data()
        pass


if __name__ == "__main__":
    main()
