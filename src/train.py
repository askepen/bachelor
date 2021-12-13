import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from model import LitModel
from data_module import CompressedAudioDataModule
from argparse import ArgumentParser


def train_from_dict(hparams, trainer_hparams):
    trainer_hparams["logger"] = (
        WandbLogger(project="Bachelor") if hparams["wandb"] else None
    )
    model = LitModel(**hparams)
    data_module = CompressedAudioDataModule(**hparams)
    trainer = pl.Trainer(**trainer_hparams)
    trainer.fit(model, data_module)


def train_from_argparse():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_fft", type=int, default=None)
    parser.add_argument("--stft_width", type=int, default=285)
    parser.add_argument("--stft_height", type=int, default=751)
    parser.add_argument("--wandb", type=bool, default=False)
    parser = Trainer.add_argparse_args(parser)
    parser = LitModel.add_model_specific_args(parser)
    args = parser.parse_args()

    model = LitModel(**vars(args))
    logger = WandbLogger(project="Bachelor") if args.wandb else None
    data_module = CompressedAudioDataModule.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train_from_argparse()
