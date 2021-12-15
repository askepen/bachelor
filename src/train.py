import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from model import LitModel
from data_module import CompressedAudioDataModule
from argparse import ArgumentParser
from logging_utils import ImagePredictionLogger


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
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--log_n_samples", type=int, default=4)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser = Trainer.add_argparse_args(parser)
    parser = CompressedAudioDataModule.add_argparse_args(parser)
    parser = LitModel.add_model_specific_args(parser)
    args = parser.parse_args()

    logger = WandbLogger(project="Bachelor") if args.wandb else None
    model = LitModel(**vars(args))
    data_module = CompressedAudioDataModule.from_argparse_args(
        args, num_workers=args.num_workers
    )
    cb_log_prediction = ImagePredictionLogger(
        args.log_n_samples, args.log_every_n_steps
    )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=cb_log_prediction
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    train_from_argparse()
