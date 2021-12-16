import pytorch_lightning as pl
import torch
from torch import nn

# from complexPyTorch.complexLayers import (
#     #     ComplexConv2d as Conv2d,
#     #     ComplexReLU as ReLU,
#     ComplexMaxPool2d as MaxPool2d,
# )
from torch.nn import Conv2d, LeakyReLU, MaxPool2d
from torchvision.transforms import CenterCrop


class LitModel(pl.LightningModule):
    def __init__(
        self,
        stft_width,
        stft_height,
        lr,
        momentum,
        num_blocks,
        optim,
        kernel_size,
        in_channels,
        out_channels,
        **kwargs,
    ):
        super().__init__()
        self.out_size = [stft_height, stft_width]
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size

        self.loss_fn = nn.MSELoss(reduction="sum")
        self.down = MaxPool2d(2, ceil_mode=True)
        self.down_blocks = torch.nn.ModuleList([
            self.block(in_channels, 64),
            self.block(64, 128),
            self.block(128, 256),
            self.block(256, 512),
        ])
        self.bottom = self.block(512, 1024)
        self.up = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.up_blocks = torch.nn.ModuleList([
            self.block(1024, 512, with_concat=True),
            self.block(512, 256, with_concat=True),
            self.block(256, 128, with_concat=True),
            self.block(128, 64, with_concat=True),
        ])
        self.out = Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1
        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument(
            "--momentum", type=float, default=0.0,
            help="Only for SGD optimizer"
        )
        parser.add_argument("--num_blocks", type=int, default=3)
        parser.add_argument("--optim", type=str, default="adam")
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--in_channels", type=int, default=2)
        parser.add_argument("--out_channels", type=int, default=2)
        return parent_parser

    def block(self, in_channels, out_channels, with_concat=False):
        in_channels = in_channels + out_channels if with_concat else in_channels
        return nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            LeakyReLU(),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            LeakyReLU(),
        )

    def crop_width_height(self, x, shape_to_match):
        return CenterCrop(shape_to_match[-2:])(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device)

        # Convert real/imag axes to channels
        x = x.permute(0, 3, 1, 2)

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

        # Convert channels to real/imag axes
        x = x.permute(0, 2, 3, 1)

        return x

    def _step(self, batch, batch_idx, step_name):
        """Generic code to run for each step in train/val/test"""
        (x, _), (y, _) = batch

        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Returns loss from single batch"""
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx) -> None:
        """Logs validation loss"""
        return self._step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        """Logs test loss"""
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self.optim == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            raise Exception(f"Optimizer '{self.optim}' not recognized")
