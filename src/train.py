import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import LitModel
from data_module import CompressedAudioDataModule
import torch


def train():
    wandb_logger = WandbLogger(project="Bachelor")
    data_module = CompressedAudioDataModule(data_dir="./data")
    model = LitModel(126, 751)
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=None,  # -1,
        max_epochs=3,
    )
    trainer.fit(model, data_module)
