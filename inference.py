import pytorch_lightning as pl
from module.data import *
from module.mrc import *
from module.retrieval import *
from datasets import load_from_disk
import hydra


@hydra.main(config_path="./config", config_name="combine", version_base=None)
def main(config):

    mode = "validation"
    top_k = 10
    only_mrc = False
    model_checkpoint = "/data/ephemeral/home/gj/level2-mrc-nlp-11/checkpoints/baseline_epoch=01_exact_match=56.25.ckpt"
    
    '''# inference.py의 15번 줄 주석 처리 후 다음 코드 추가
    # 1. 체크포인트 디렉토리 경로 예시
    checkpoints_dir = "/data/ephemeral/home/level2-mrc-nlp-11/checkpoints" # 내 체크포인트 폴더의 경로로 변경하기
    # 2. 최신 체크포인트 파일을 찾음
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.ckpt"))  # 모든 .ckpt 파일 찾기
    if checkpoint_files:
        # 가장 최근 체크포인트 파일
        model_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Using model checkpoint: {model_checkpoint}")
    else: # 3.체크 포인트가 존재하지 않으면 에러가 발생됨
        raise FileNotFoundError("No checkpoint files found in the specified directory.")
    '''
    

    if mode == "validation":
        # 1. load eval examples
        dataset_list = get_dataset_list(config.mrc.data.dataset_name)
        eval_examples = concatenate_datasets([ds["validation"] for ds in dataset_list])

        if not only_mrc:
            # 2. retrieve context
            retrieval = TfIdfRetrieval(config.retrieval)
            retrieval.fit()
            retrieval.create_embedding_vector()
            doc_ids, docs = retrieval.search(eval_examples["question"], k=top_k)

            # 3. change original context to retrieved context
            eval_examples = eval_examples.remove_columns(["context"])
            eval_examples = eval_examples.add_column(
                "context", [" ".join(doc) for doc in docs]
            )

        # 4. make eval_dataset & eval_dataloader
        data_module = MrcDataModule(config.mrc)
        data_module.eval_dataset, preproc_eval_examples = data_module.get_dataset(
            eval_examples
        )
        val_dataloader = data_module.val_dataloader()

        # 5. load mrc model
        mrc = MrcLightningModule.load_from_checkpoint(model_checkpoint)
        # 5.1. put eval examples and eval dataset to inference
        mrc.eval_examples = preproc_eval_examples
        mrc.eval_dataset = data_module.eval_dataset

        # 6. inference eval dataset
        trainer = pl.Trainer()
        trainer.validate(model=mrc, dataloaders=val_dataloader)

    else:
        # 1. load test examples
        if not os.path.exists(f"./data/test_dataset/"):
            get_dataset_list(["default"])
        test_examples = load_from_disk("./data/test_dataset/")["validation"]

        # 2. retrieve context
        retrieval = TfIdfRetrieval(config.retrieval)
        retrieval.fit()
        retrieval.create_embedding_vector()
        doc_ids, docs = retrieval.search(test_examples["question"], k=top_k)

        # 3. insert retrieved context column
        test_examples = test_examples.add_column(
            "context", [" ".join(doc) for doc in docs]
        )

        # 4. make eval_dataset & eval_dataloader
        data_module = MrcDataModule(config.mrc)
        data_module.test_dataset, preproc_test_examples = data_module.get_dataset(
            test_examples
        )
        test_dataloader = data_module.test_dataloader()

        # 5. load mrc model
        mrc = MrcLightningModule.load_from_checkpoint(model_checkpoint)
        # 5.1. put eval examples and eval dataset to inference
        mrc.test_examples = preproc_test_examples
        mrc.test_dataset = data_module.test_dataset

        # 6. inference eval dataset
        trainer = pl.Trainer()
        trainer.test(model=mrc, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
