import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import module.dense_retrieval_data as module_data
from module.dense_retrieval_model import (
    BiEncoderDenseRetrieval,
    CrossEncoderDenseRetrieval,
)
import module.dense_retrieval_encoder as module_encoder
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import set_caching_enabled
import hydra
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

set_caching_enabled(False)


@hydra.main(config_path="./config", config_name="retrieval", version_base=None)
def main(config):
    mode = "bi"

    config = config.bi if mode == "bi" else config.cross
    # 0. logger
    logger = WandbLogger(project=config.wandb.project) if config.wandb.enable else None

    # 1. set data_module(=pl.LightningDataModule class)
    data_module = getattr(module_data, config.data.data_module)(config)

    # 2. set model
    # 2.1. set retrieval module(=pl.LightningModule class)
    retrieval = (
        BiEncoderDenseRetrieval(config)
        if mode == "bi"
        else CrossEncoderDenseRetrieval(config)
    )

    # 3. set trainer(=pl.Trainer) & train
    run_name = f"{mode}-encoder={config.model.plm_name}_{config.data.neg_sampling_method}_bz={config.data.batch_size}_lr={config.optimizer.lr}".replace(
        "/", "-"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=run_name + "_{epoch:02d}-{accuracy:.2f}",
        save_top_k=1,
        monitor="accuracy",
        mode="max",
    )
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        val_check_interval=0.5,
        accelerator="cuda",
        devices=1,
        max_epochs=config.train.num_train_epochs,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        logger=logger,
    )
    trainer.fit(model=retrieval, datamodule=data_module)

    # 4. test by validation dataset
    # retrieval.create_embedding_vector()
    # print(retrieval.search("우리나라 대통령은 누구야?"))


if __name__ == "__main__":
    main()
