import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from models import LitUnet, LitCNN, LitFullyConnected
from data_module import CompressedAudioDataModule
from argparse import ArgumentParser, Namespace
from logging_utils import ImagePredictionLogger

MODEL = LitFullyConnected


def train_from_dict(args_dict):
    args = Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    train(args)


def train_from_argparse():
    parser = ArgumentParser()
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--log_n_samples", type=int, default=4)
    parser.add_argument("--log_prediction_freq", type=int, default=4)
    parser = Trainer.add_argparse_args(parser)
    parser = CompressedAudioDataModule.add_argparse_args(parser)
    parser = MODEL.add_model_specific_args(parser)
    args = parser.parse_args()
    train(args)


def train(args: Namespace):
    seed_everything(1337)
    model = MODEL(**vars(args))
    logger = WandbLogger(project="Bachelor") if args.wandb else None
    # logger.watch(model, log_freq=500)
    cb_log_prediction = ImagePredictionLogger(
        args.log_n_samples, args.log_prediction_freq
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger,  callbacks=cb_log_prediction
    )
    data_module = CompressedAudioDataModule.from_argparse_args(args)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    train_from_argparse()
