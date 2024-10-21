import hydra
from dotenv import load_dotenv
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import set_caching_enabled

from module.mrc import *
import module.data as module_data

# .env 파일 로드
load_dotenv()

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

set_caching_enabled(False)


@hydra.main(config_path="./config", config_name="mrc", version_base=None)
def main(config):
    # 0. logger
    logger = WandbLogger(project=config.wandb.project) if config.wandb.enable else None

    # 1. set data_module(=pl.LightningDataModule class)
    data_module = getattr(module_data, config.data.data_module)(config)
    data_module.setup()
    # 1.1. create eval & test dataset
    eval_dataset = data_module.eval_dataset
    test_dataset = data_module.test_dataset
    eval_examples = data_module.eval_examples
    test_examples = data_module.test_examples

    # 2. set mrc module(=pl.LightningModule class)
    model_module = MrcLightningModule(
        config, eval_dataset, test_dataset, eval_examples, test_examples
    )

    # 3. set trainer(=pl.Trainer) & train
    if config.wandb.enable and logger is not None:
        run_name = f"{logger.experiment.name}_{config.data.preproc_list[0]}_{config.data.dataset_name[0]}_bz={config.data.batch_size}_lr={config.optimizer.lr}"
    else:
        run_name = f"default_{config.data.preproc_list[0]}_{config.data.dataset_name[0]}_bz={config.data.batch_size}_lr={config.optimizer.lr}"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=run_name + "_baseline_{epoch:02d}_{exact_match:.2f}",
        save_top_k=1,
        monitor="exact_match",
        mode="max",
    )
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.num_train_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        logger=logger,
        precision="16-mixed",
    )
    trainer.fit(model=model_module, datamodule=data_module)


if __name__ == "__main__":
    main()
