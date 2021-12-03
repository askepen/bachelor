import pytorch_lightning as pl
import torch
from torch import nn
from complexPyTorch.complexLayers import (
    #     ComplexConv2d as Conv2d,
    #     ComplexReLU as ReLU,
    ComplexMaxPool2d as MaxPool2d,
)
from torch.nn import Conv2d, ReLU  # , MaxPool2d


class LitModel(pl.LightningModule):
    def block(self, in_channels, out_channels):
        return nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ReLU(),
        )

    def __init__(self, in_features: int, out_features: int, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.MSELoss(reduction="sum")

        self.down = MaxPool2d((2, 2))
        self.down_blocks = [
            self.block(in_features, 64),
            self.block(64, 128),
            self.block(128, 256),
            self.block(256, 512),
        ]

        self.bottom = self.block(512, 1024)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_blocks = [
            self.block(1024, 512),
            self.block(512, 256),
            self.block(256, 128),
            self.block(128, 64),
        ]

        self.out = Conv2d(
            in_channels=64, out_channels=out_features, kernel_size=1)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("\nFORWARD\n")
        skip = []

        # Contracting path
        for block in self.down_blocks:
            print(x.shape)
            print("block")
            x = block(x)
            print(x.shape)
            print()

            print(x.shape)
            print("append skip")
            skip.append(x)
            print(x.shape)
            print()

            x = torch.view_as_complex(x)
            print(x.shape)
            print("down")
            x = self.down(x)
            x = torch.view_as_real(x)
            print(x.shape)
            print()

        x = self.bottom(x)

        # Expanding path
        for skip_connection, block in zip(skip, self.up_blocks):
            x = self.up(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = block(x)

        x = self.out(x)

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
        return self._step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        """Logs test loss"""
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
