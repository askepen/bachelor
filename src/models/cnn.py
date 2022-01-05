import pytorch_lightning as pl
import torch
from torch import nn

# from complexPyTorch.complexLayers import (
#     #     ComplexConv2d as Conv2d,
#     #     ComplexReLU as ReLU,
#     ComplexMaxPool2d as MaxPool2d,
# )
from torch.nn import Conv2d, LeakyReLU, MaxPool2d
from torch.nn.modules.upsampling import UpsamplingBilinear2d
from torchvision.transforms import CenterCrop
import loss


class LitCNN(pl.LightningModule):
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
        mid_channels,
        out_channels,
        **kwargs,
    ):
        super().__init__()
        self.out_size = [stft_height, stft_width]
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.kernel_size = kernel_size

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = loss.MSLELoss()

        self.first = Conv2d(in_channels, mid_channels, 1)
        self.layers = nn.Sequential(
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            LeakyReLU(),
            UpsamplingBilinear2d(scale_factor=1.35),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            LeakyReLU(),
            UpsamplingBilinear2d(scale_factor=1.35),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            LeakyReLU(),
            UpsamplingBilinear2d(scale_factor=1.35),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            LeakyReLU(),
            UpsamplingBilinear2d(scale_factor=1.35),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            LeakyReLU(),
            UpsamplingBilinear2d(scale_factor=1.35),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            LeakyReLU(),
            UpsamplingBilinear2d(scale_factor=1.35),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            nn.ReLU(),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            nn.ReLU(),
            Conv2d(
                mid_channels, mid_channels, self.kernel_size,
                padding="same",
                padding_mode="reflect",
            ),
            nn.ReLU(),
        )
        self.last = Conv2d(mid_channels, out_channels, 1)

        self.save_hyperparameters()

    @ staticmethod
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
        parser.add_argument("--mid_channels", type=int, default=32)
        parser.add_argument("--out_channels", type=int, default=2)
        return parent_parser

    def crop_width_height(self, x, shape_to_match):
        return CenterCrop(shape_to_match[-2:])(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device)

        x = torch.view_as_complex(x)
        phase, x = torch.angle(x), torch.abs(x)
        x = x.unsqueeze(1)

        # x = x.permute(0, 3, 1, 2)

        x = self.first(x)
        x = self.layers(x)
        x = self.last(x)
        x = self.crop_width_height(x, self.out_size)

        # x = x.permute(0, 2, 3, 1)

        x = x.squeeze(1)
        x = torch.polar(x, phase)
        x = torch.view_as_real(x)

        return x

    def _step(self, batch, batch_idx, step_name):
        """Generic code to run for each step in train/val/test"""
        (x, _), (y, _) = batch
        pred = self(x)
        loss = torch.sqrt(self.loss_fn(pred, y))
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx) -> None:
        return self._step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self.optim == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            raise Exception(f"Optimizer '{self.optim}' not recognized")
