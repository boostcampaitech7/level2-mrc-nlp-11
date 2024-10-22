import torch
import numpy as np
import hydra
from dotenv import load_dotenv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets import disable_caching

import module.data as module_data
from module.retrieval import BiEncoderDenseRetrieval

# .env 파일 로드
load_dotenv()

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

disable_caching()


@hydra.main(config_path="./config", config_name="retrieval", version_base=None)
def main(config):

    config = config.bi
    # 0. logger
    if config.model.use_lora:
        run_name = f"lora-bi-encoder={config.model.plm_name}_use-overflow-token={1 if not config.data.use_overflow_token else config.data.overflow_limit}_num-neg={config.data.num_neg}_bz={config.data.batch_size}_lr={config.optimizer.lr}".replace(
            "/", "-"
        )
    else:
        run_name = f"bi-encoder={config.model.plm_name}_use-overflow-token={1 if not config.data.use_overflow_token else config.data.overflow_limit}_num-neg={config.data.num_neg}_bz={config.data.batch_size}_lr={config.optimizer.lr}".replace(
            "/", "-"
        )

    logger = (
        WandbLogger(name=run_name, project=config.wandb.project)
        if config.wandb.enable
        else None
    )

    # 1. set data_module(=pl.LightningDataModule class)
    data_module = getattr(module_data, config.data.data_module)(config)

    # 2. set model
    # 2.1. set retrieval module(=pl.LightningModule class)
    retrieval = BiEncoderDenseRetrieval(config)

    # 3. set trainer(=pl.Trainer) & train
    checkpoint_callback = ModelCheckpoint(
        dirpath="retrieval_checkpoints",
        filename=run_name + "_{epoch:02d}_{accuracy:.2f}",
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

    # 4. save last model
    trainer.save_checkpoint(f"./retrieval_checkpoints/{run_name}.ckpt")


if __name__ == "__main__":
    main()
