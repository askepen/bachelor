import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import LitModel
from data_module import CompressedAudioDataModule


def train():
    wandb_logger = WandbLogger(project="Bachelor")
    data_module = CompressedAudioDataModule(data_dir="./data")
    model = LitModel(126, 751)
    trainer = pl.Trainer(
        logger=wandb_logger,
        # gpus=-1,
        progress_bar_refresh_rate=20,
        tpu_cores=1,
        max_epochs=3,
    )
    trainer.fit(model, data_module)
