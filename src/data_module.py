import os
import transforms
from torch import nn
from pytorch_lightning import LightningDataModule
from dataset import CompressedAudioDataset
from torch.utils.data import DataLoader, random_split


class CompressedAudioDataModule(LightningDataModule):
    """PyTorch-Lightning data module for the compressed audio dataset"""

    def __init__(self, data_dir, batch_size, n_fft, stft_width, stft_height, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()
        self.transform = nn.Sequential(
            transforms.RandomSubsample(),
            transforms.STFT(n_fft),
            transforms.PadToSize([stft_height, stft_width]),
            transforms.ViewAsReal(),
        )

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
                self.data_dir, train=True, transform=self.transform
            )
            lengths = [round(x * len(dataset_train)) for x in [0.25, 0.75]]
            self.dataset_train, self.dataset_val = random_split(dataset_train, lengths)
        if stage == "test" or stage is None:
            self.dataset_test = CompressedAudioDataset(
                self.data_dir, train=False, transform=self.transform
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
