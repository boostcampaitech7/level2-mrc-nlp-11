import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.data import MrcDataModule
from module.retrieval import TfIdfRetrieval
import hydra
import json
import pandas as pd


def save_tokenized_samples(tokenizer, dataset, output_dir, prefix):

    tokenized_context = {
        dataset[i]["example_id"]: tokenizer.convert_ids_to_tokens(
            dataset[i]["input_ids"]
        )
        for i in range(len(dataset))
    }

    token_file = os.path.join(output_dir, f"{prefix}_tokenized_samples.json")

    with open(token_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(tokenized_context, indent=4, ensure_ascii=False) + "\n")


def save_wiki_tfidf_info(retriever, output_dir):
    retriever.fit()
    retriever.create_embedding_vector()
    dense_matrix = retriever.sparse_embedding_matrix.toarray()
    feature_names = retriever.vectorizer.get_feature_names_out()
    df = pd.DataFrame(dense_matrix, columns=feature_names)

    tfidf_info = {
        i: {token: df.iloc[i][token] for token in df.iloc[i].nlargest(10).index}
        for i in range(len(df))
    }

    wiki_tfidf_info_file = os.path.join(output_dir, f"wiki_tfidf_info.json")

    with open(wiki_tfidf_info_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(tfidf_info, indent=4, ensure_ascii=False) + "\n")


# train.py에서 config.tokenizer.save = False로 해놔서 토큰 저장 못 했으면 이 파일을 직접 실행!
@hydra.main(config_path="../config", config_name="combine", version_base=None)
def main(config):
    train_token_file = os.path.join(
        config.mrc.train.output_dir, f"train_tokenized_samples.json"
    )
    eval_token_file = os.path.join(
        config.mrc.train.output_dir, f"eval_tokenized_samples.json"
    )
    wiki_tfidf_info_file = os.path.join(
        config.mrc.train.output_dir, f"wiki_tfidf_info.json"
    )
    if not (os.path.exists(train_token_file) and os.path.exists(eval_token_file)):
        print("==================================================")
        print("train/eval dataset의 tokenized_sample을 생성합니다.")
        print("==================================================")

        data_module = MrcDataModule(config.mrc)
        data_module.setup()

        # eval_tokenized_samples
        save_tokenized_samples(
            data_module.tokenizer,
            data_module.eval_dataset,
            config.mrc.train.output_dir,
            "eval",
        )
        # train_tokenized_samples
        save_tokenized_samples(
            data_module.tokenizer,
            data_module.train_dataset,
            config.mrc.train.output_dir,
            "train",
        )

        print("=====================================================")
        print("train/eval dataset의 tokenized_sample가 생성되었습니다.")
        print("=====================================================")

    if not (os.path.exists(wiki_tfidf_info_file)):
        print("==================================================")
        print("wiki documents의 tfidf 점수 정보 데이터를 생성합니다.")
        print("==================================================")

        retriever = TfIdfRetrieval(config.retrieval)
        save_wiki_tfidf_info(retriever, config.mrc.train.output_dir)

        print("======================================================")
        print("wiki documents의 tfidf 점수 정보 데이터가 생성되었습니다.")
        print("======================================================")


if __name__ == "__main__":
    main()
