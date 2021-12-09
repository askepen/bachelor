import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import LitModel
from data_module import CompressedAudioDataModule
from argparse import ArgumentParser


def train():
    parser = ArgumentParser()
    parser.add_argument("batch_size")
    parser.add_argument("learning_rate")
    parser.add_argument("gpus")
    parser.add_argument("tpus")

    wandb_logger = WandbLogger(project="Bachelor")
    data_module = CompressedAudioDataModule(data_dir="./data", batch_size=2)

    model = LitModel(out_size=[751, 285])
    trainer = pl.Trainer(
        # logger=wandb_logger,
        gpus=None,
        progress_bar_refresh_rate=1,
        # tpu_cores=1,
        max_epochs=3,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
