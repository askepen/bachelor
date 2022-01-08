import os

from pytorch_lightning.utilities.argparse import add_argparse_args
import transforms
from torch import nn
from pytorch_lightning import LightningDataModule
from dataset import CompressedAudioDataset
from torch.utils.data import DataLoader, random_split


class CompressedAudioDataModule(LightningDataModule):
    """PyTorch-Lightning data module for the compressed audio dataset"""

    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        n_fft,
        stft_width,
        stft_height,
        trim_left,
        train_set_fraction,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set_fraction = train_set_fraction
        self.transform = nn.Sequential(
            transforms.Trim(left=trim_left),
            transforms.BSpline(15_000),
            transforms.TrimToSize(sample_length=48_000),
            # transforms.STFT(n_fft),
            # transforms.PadToSize([stft_height, stft_width]),
            # transforms.ViewAsReal(),
            transforms.DropSampleRate(),
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("CompressedAudioDataModule")
        parser.add_argument("--data_dir", type=str, default="./data")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=os.cpu_count())
        parser.add_argument("--n_fft", type=int, default=None)
        parser.add_argument("--stft_width", type=int, default=285)
        parser.add_argument("--stft_height", type=int, default=751)
        parser.add_argument("--train_set_fraction", type=float, default=0.8)
        parser.add_argument("--trim_left", type=float, default=0.0)
        parser.add_argument("--trim_right", type=float, default=0.0)
        return parent_parser

    def prepare_data(self):
        """Makes sure we have downloaded and processed the data. Is only called once and on 1 GPU"""
        CompressedAudioDataset(self.data_dir, train=True)
        CompressedAudioDataset(self.data_dir, train=False)

    def setup(self, stage=None):
        """
        We set up only relevant datasets when stage is specified (stage is provided by Pytorch-Lightning).
        Called one each GPU separately - stage defines if we are at fit or test step
        """
        if stage == "fit" or stage is None:
            dataset_train = CompressedAudioDataset(
                self.data_dir, train=True, transform=self.transform, use_baked_data=False,
            )
            lengths = [
                round(self.train_set_fraction * len(dataset_train)),
                round((1.0 - self.train_set_fraction) * len(dataset_train)),
            ]
            self.dataset_train, self.dataset_val = random_split(
                dataset_train, lengths
            )
        if stage == "test" or stage is None:
            self.dataset_test = CompressedAudioDataset(
                self.data_dir, train=False, transform=self.transform, device=self.device
            )

    def train_dataloader(self):
        """Returns training dataloader"""
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Returns validation dataloader"""
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Returns test dataloader"""
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
