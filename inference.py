import os, glob, copy
from dotenv import load_dotenv
import pytorch_lightning as pl

from module.data import *
from module.mrc import *
from module.retrieval import *
from datasets import load_from_disk, Dataset

# .env 파일 로드
load_dotenv()


def main(
    mrc_checkpoint=None,
    sparse_retrieval_checkpoint=None,
    dense_retrieval_checkpoint=None,
    run_mrc=True,
    run_retrieval=False,
    top_k=10,
    mode="validation",
    use_separate_inference=False,
):

    # 0. load model
    # 0.1. load mrc model
    if run_mrc:
        mrc = MrcLightningModule.load_from_checkpoint(mrc_checkpoint)
        mrc.inference_mode = "separate" if use_separate_inference else None
    # 0.2. load retrieval model
    if run_retrieval:
        with open(sparse_retrieval_checkpoint, "rb") as file:
            retrieval = pickle.load(file)
        if dense_retrieval_checkpoint:
            dense_retrieval = BiEncoderDenseRetrieval.load_from_checkpoint(
                dense_retrieval_checkpoint
            )
            retrieval = RetrievalReranker(
                sparse_retrieval=retrieval, dense_retrieval=dense_retrieval
            )

    # 0.3. set config
    config = mrc.config if run_mrc else None

    if mode == "validation":
        # 1. load eval examples
        dataset_list = get_dataset_list(
            config.data.dataset_name if config else ["default"]
        )
        eval_examples = concatenate_datasets([ds["validation"] for ds in dataset_list])

        if run_retrieval:
            # 2. retrieve context
            if not dense_retrieval_checkpoint:
                # 2.1. not use rerank
                docs_score, docs_idx, docs, titles = retrieval.search(
                    eval_examples["question"], k=top_k
                )
            else:
                # 2.2. use rerank
                docs_score, docs_idx, docs, titles = [], [], [], []
                for question, context in zip(
                    eval_examples["question"], eval_examples["context"]
                ):
                    score, idx, doc, title = retrieval.search(question)
                    docs_score.append(score)
                    docs_idx.append(idx)
                    docs.append(doc)
                    titles.append(title)

            # 3. calculate retrieval accuracy
            cnt = 0
            for context, doc in zip(eval_examples["context"], docs):
                if context.replace(" ", "").replace("\n", "").replace("\\n", "") in [
                    d.replace(" ", "").replace("\n", "").replace("\\n", "") for d in doc
                ]:
                    cnt += 1
            print(f"validation retrieval, total: {len(eval_examples)}, correct: {cnt}")

            # 4. preprocess retrieved context
            if config and "title_context_merge_token" in config.data.preproc_list:
                docs = [
                    [
                        f"<TITLE> {titles[i][j]} <TITLE_END> {docs[i][j]}"
                        for j in range(len(docs[i]))
                    ]
                    for i in range(len(docs))
                ]

            # 5. change original context to retrieved context in eval_examples
            eval_examples = eval_examples.remove_columns(["context"])
            # 5-1. separate inference
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
            # 5-2. concat inference
            else:
                eval_examples = eval_examples.add_column(
                    "context", [" ".join(doc) for doc in docs]
                )

        if not run_mrc:
            return

        # 6. make eval_dataset & eval_dataloader
        data_module = MrcDataModule(config)
        data_module.eval_dataset, preproc_eval_examples = data_module.get_dataset(
            eval_examples
        )
        val_dataloader = data_module.val_dataloader()

        # 6.1. put eval examples and eval dataset to inference
        mrc.eval_examples = preproc_eval_examples
        mrc.eval_dataset = data_module.eval_dataset

        # 7. inference eval dataset
        trainer = pl.Trainer()
        trainer.validate(model=mrc, dataloaders=val_dataloader)

    else:
        # 1. load test examples
        if not os.path.exists(f"./data/test_dataset/"):
            get_dataset_list(["default"])
        test_examples = load_from_disk("./data/test_dataset/")["validation"]

        if not run_retrieval:
            return

        # 2. retrieve context
        if not dense_retrieval_checkpoint:
            # 2.1. not use rerank
            docs_score, docs_idx, docs, titles = retrieval.search(
                eval_examples["question"], k=top_k
            )
        else:
            # 2.2. use rerank
            docs_score, docs_idx, docs, titles = [], [], [], []
            for question, context in zip(
                eval_examples["question"], eval_examples["context"]
            ):
                score, idx, doc, title = retrieval.search(question)
                docs_score.append(score)
                docs_idx.append(idx)
                docs.append(doc)
                titles.append(title)

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

        if not run_mrc:
            return

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
    # sparse model checkpoint 경로 (없으면 None으로 설정하세요)
    sparse_retrieval_checkpoint = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/retrieval_checkpoints/combine-bm25_model=BM25Okapi_analyzer=Kkma_tokenizer=klue-bert-base"
    )
    # dense model checkpoint 경로 (없으면 None으로 설정하세요)
    dense_retrieval_checkpoint = (
        os.getenv("DIR_PATH")
        + "/level2-mrc-nlp-11/retrieval_checkpoints/lora-bi-encoder=klue-roberta-base_use-overflow-token=True_num-neg=8_bz=4_lr=2e-05_epoch=14-accuracy=0.85_emb-vec.ckpt"
    )
    # mrc model checkpoint 경로 (없으면 None으로 설정하세요)
    mrc_checkpoint = "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/checkpoints/fine_tuned/klue-roberta-large_korquad1.0_filtered_classic-valley-65_original_default_bz=16_lr=1.6764783497920226e-05_fine_tuned_epoch=03_exact_match=71.67.ckpt"

    # main() 인자
    mode = "validation"
    top_k = 10
    run_mrc = True
    run_retrieval = True
    use_separate_inference = False  # Separate Inference 사용 시 True로 설정

    # 1. 체크포인트 디렉토리 경로 예시
    checkpoints_dir = (
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/checkpoints"
    )  # 내 체크포인트 폴더의 경로로 변경하기
    # 2. 최신 체크포인트 파일을 찾음
    checkpoint_files = glob.glob(
        os.path.join(checkpoints_dir, "*.ckpt")
    )  # 모든 .ckpt 파일 찾기
    if not mrc_checkpoint and checkpoint_files:
        # 가장 최근 체크포인트 파일
        mrc_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Using mrc model checkpoint: {mrc_checkpoint}")

    main(
        mrc_checkpoint,
        sparse_retrieval_checkpoint,
        dense_retrieval_checkpoint,
        run_mrc,
        run_retrieval,
        top_k,
        mode,
        use_separate_inference,
    )
