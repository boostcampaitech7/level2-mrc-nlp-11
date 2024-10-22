import glob
import pytorch_lightning as pl
from module.data import *
from module.mrc import *
from module.retrieval import *
from datasets import load_from_disk, Dataset
import os
from dotenv import load_dotenv
import copy

# .env 파일 로드
load_dotenv()


def main(retrieval_checkpooint, mrc_checkpoint):

    mode = "validation"
    top_k = 10
    only_mrc = False
    # Separate Inference: top-k 문서를 합치지 않고 따로따로 inference
    use_separate_inference = True

    # 0. load model
    # 0.1. load mrc model
    mrc = MrcLightningModule.load_from_checkpoint(mrc_checkpoint)
    mrc.inference_mode = "separate" if use_separate_inference else None
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
            # 3-1. separate inference
            eval_examples = eval_examples.remove_columns(["context"])
            if use_separate_inference:
                separate_eval_examples = []
                copied_eval_examples = copy.deepcopy(eval_examples)
                eval_examples = None
                for i, eval_example in enumerate(copied_eval_examples):
                    for k in range(top_k):
                        new_eval_example = copy.deepcopy(eval_example)
                        new_eval_example["context"] = docs[i][k]
                        new_eval_example["doc_score"] = docs_score[i][k]
                        new_eval_example["document_id"] = docs_idx[i][k]
                        new_eval_example["id"] = new_eval_example["id"] + f"_top{k}"
                        separate_eval_examples.append(new_eval_example)
                eval_examples = Dataset.from_list(separate_eval_examples)
            # 3-2. concat inference
            else:
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
        # 3-1. separate inference
        if use_separate_inference:
            separate_test_examples = []
            copied_test_examples = copy.deepcopy(test_examples)
            test_examples = None
            for i, test_example in enumerate(copied_test_examples):
                for k in range(top_k):
                    new_test_example = copy.deepcopy(test_example)
                    new_test_example["context"] = docs[i][k]
                    new_test_example["doc_score"] = docs_score[i][k]
                    new_test_example["document_id"] = docs_idx[i][k]
                    new_test_example["id"] = new_test_example["id"] + f"_top{k}"
                    separate_test_examples.append(new_test_example)
            test_examples = Dataset.from_list(separate_test_examples)
        # 3-2. concat inference
        else:
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
        + "/level2-mrc-nlp-11/retrieval_checkpoints/combine-bm25_model=BM25Okapi_analyzer=Kkma_tokenizer=klue-bert-base"
    )
    # # 1. 체크포인트 디렉토리 경로 예시
    # checkpoints_dir = (
    #     os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/checkpoints"
    # )  # 내 체크포인트 폴더의 경로로 변경하기
    # # 2. 최신 체크포인트 파일을 찾음
    # checkpoint_files = glob.glob(
    #     os.path.join(checkpoints_dir, "*.ckpt")
    # )  # 모든 .ckpt 파일 찾기
    # if checkpoint_files:
    #     # 가장 최근 체크포인트 파일
    #     mrc_checkpoint = max(checkpoint_files, key=os.path.getctime)
    #     print(f"Using mrc model checkpoint: {mrc_checkpoint}")
    # else:  # 3.체크 포인트가 존재하지 않으면 에러가 발생됨
    #     raise FileNotFoundError("No checkpoint files found in the specified directory.")
    mrc_checkpoint = "/data/ephemeral/home/jaehyeop/level2-mrc-nlp-11/checkpoints/driven-wildflower-57_original_default_bz=16_lr=1.7110631470130408e-05_baseline_epoch=00_exact_match=71.25.ckpt"

    main(retrieval_checkpoint, mrc_checkpoint)
