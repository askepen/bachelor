import pytorch_lightning as pl
import torch
from torch import nn
# from complexPyTorch.complexLayers import (
#     #     ComplexConv2d as Conv2d,
#     #     ComplexReLU as ReLU,
#     ComplexMaxPool2d as MaxPool2d,
# )
from torch.nn import Conv2d, ReLU, MaxPool2d
from torchvision.transforms import CenterCrop


class LitModel(pl.LightningModule):
    def __init__(self, stft_width, stft_height_out, lr, **kwargs):
        super().__init__()
        self.out_size = [stft_height_out, stft_width]
        self.lr = lr
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.down = MaxPool2d(2, ceil_mode=True)
        self.down_blocks = [
            self.block(2, 64),
            self.block(64, 128),
            self.block(128, 256),
            self.block(256, 512),
        ]
        self.bottom = self.block(512, 1024)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_blocks = [
            self.block(1024, 512, with_concat=True),
            self.block(512, 256, with_concat=True),
            self.block(256, 128, with_concat=True),
            self.block(128, 64, with_concat=True),
        ]
        self.out = Conv2d(in_channels=64, out_channels=2, kernel_size=1)

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--num_blocks", type=int, default=4)
        return parent_parser

    def block(self, in_channels, out_channels, with_concat=False):
        in_channels = in_channels + out_channels if with_concat else in_channels
        return nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ReLU(),
        )

    def crop_width_height(self, x, shape_to_match):
        return CenterCrop(shape_to_match[-2:])(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat real/imag axes as channels
        # x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        # Make output shape match input shape if it not specified
        self.out_size = self.out_size or x.shape

        skip = []

        for block in self.down_blocks:
            x = block(x)
            skip.append(x)
            x = self.down(x)

        x = self.bottom(x)

        for skip_connection, block in zip(skip[::-1], self.up_blocks):
            x = self.up(x)
            skip_connection = self.crop_width_height(skip_connection, x.shape)
            x = torch.cat((x, skip_connection), dim=1)
            x = block(x)

        x = self.out(x)
        x = self.crop_width_height(x, self.out_size)

        # Treat channels to real/imag axes
        x = x.permute(0, 2, 3, 1)
        # x = x.reshape(x.permute(0, 2, 3, 1).shape)

        return x

    def _step(self, batch, batch_idx, step_name):
        """Generic code to run for each step in train/val/test"""
        (x, x_sr), (y, y_sr) = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Returns loss from single batch"""
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx) -> None:
        """Logs validation loss"""
        loss = self._step(batch, batch_idx, "valid")
        # n_fft = sample_rate // (2 ** 5)
        # audio_utils.plot_specgram(
        #     spec, sample_rate, n_fft=n_fft,
        #     ylim_freq=None, n_yticks=13, title=title
        # )
        return loss

    def test_step(self, batch, batch_idx):
        """Logs test loss"""
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
