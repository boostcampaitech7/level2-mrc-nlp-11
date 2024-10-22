import glob
import pytorch_lightning as pl
from module.data import *
from module.mrc import *
from module.retrieval import *
from datasets import load_from_disk
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def main(retrieval_checkpooint, mrc_checkpoint):

    mode = "validation"
    top_k = 10
    only_mrc = False

    # 0. load model
    # 0.1. load mrc model
    mrc = MrcLightningModule.load_from_checkpoint(mrc_checkpoint)
    # 0.2. load retrieval model
    if not only_mrc:
        with open(retrieval_checkpooint, "rb") as file:
            retrieval = pickle.load(file)
    # 0.3. set config
    config = mrc.config

    if mode == "validation":
        # 1. load eval examples
        dataset_list = get_dataset_list(config.data.dataset_name)
        eval_examples = concatenate_datasets([ds["validation"] for ds in dataset_list])

        if not only_mrc:
            # 2. retrieve context
            docs_score, docs_idx, docs, titles = retrieval.search(
                eval_examples["question"], k=top_k
            )

            if "title_context_merge_token" in config.data.preproc_list:
                docs = [
                    [
                        f"<TITLE> {titles[i][j]} <TITLE_END> {docs[i][j]}"
                        for j in range(len(docs[i]))
                    ]
                    for i in range(len(docs))
                ]

            # 3. change original context to retrieved context
            eval_examples = eval_examples.remove_columns(["context"])
            eval_examples = eval_examples.add_column(
                "context", [" ".join(doc) for doc in docs]
            )

        # 4. make eval_dataset & eval_dataloader
        data_module = MrcDataModule(config)
        data_module.eval_dataset, preproc_eval_examples = data_module.get_dataset(
            eval_examples
        )
        val_dataloader = data_module.val_dataloader()

        # 4.1. put eval examples and eval dataset to inference
        mrc.eval_examples = preproc_eval_examples
        mrc.eval_dataset = data_module.eval_dataset

        # 5. inference eval dataset
        trainer = pl.Trainer()
        trainer.validate(model=mrc, dataloaders=val_dataloader)

    else:
        # 1. load test examples
        if not os.path.exists(f"./data/test_dataset/"):
            get_dataset_list(["default"])
        test_examples = load_from_disk("./data/test_dataset/")["validation"]

        # 2. retrieve context
        docs_score, docs_idx, docs, titles = retrieval.search(
            test_examples["question"], k=top_k
        )

        if "title_context_merge_token" in config.data.preproc_list:
            docs = [
                [
                    f"<TITLE> {titles[i][j]} <TITLE_END> {docs[i][j]}"
                    for j in range(len(docs[i]))
                ]
                for i in range(len(docs))
            ]

        # 3. insert retrieved context column
        test_examples = test_examples.add_column(
            "context", [" ".join(doc) for doc in docs]
        )

        # 4. make eval_dataset & eval_dataloader
        data_module = MrcDataModule(config)
        data_module.test_dataset, preproc_test_examples = data_module.get_dataset(
            test_examples
        )
        test_dataloader = data_module.test_dataloader()

        # 4.1. put eval examples and eval dataset to inference
        mrc.test_examples = preproc_test_examples
        mrc.test_dataset = data_module.test_dataset

        # 5. inference eval dataset
        trainer = pl.Trainer()
        trainer.test(model=mrc, dataloaders=test_dataloader)


if __name__ == "__main__":
    retrieval_checkpoint = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/retrieval_checkpoints/tf-idf_tokenizer=klue-bert-base_ngram=[1, 1]"
    )
    # 1. 체크포인트 디렉토리 경로 예시
    checkpoints_dir = (
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/checkpoints"
    )  # 내 체크포인트 폴더의 경로로 변경하기
    # 2. 최신 체크포인트 파일을 찾음
    checkpoint_files = glob.glob(
        os.path.join(checkpoints_dir, "*.ckpt")
    )  # 모든 .ckpt 파일 찾기
    if checkpoint_files:
        # 가장 최근 체크포인트 파일
        mrc_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Using mrc model checkpoint: {mrc_checkpoint}")
    else:  # 3.체크 포인트가 존재하지 않으면 에러가 발생됨
        raise FileNotFoundError("No checkpoint files found in the specified directory.")

    main(retrieval_checkpoint, mrc_checkpoint)
