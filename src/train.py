import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import LitModel
from data_module import CompressedAudioDataModule
from argparse import ArgumentParser


def train():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--tpu_cores", type=int, default=None)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=1)
    parser = LitModel.add_model_specific_args(parser)
    args = parser.parse_args()

    logger = WandbLogger(project="Bachelor") if args.wandb else None
    data_module = CompressedAudioDataModule.from_argparse_args(args)
    model = LitModel(out_size=[751, 285])
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
