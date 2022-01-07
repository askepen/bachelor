import pytorch_lightning as pl
import torch
from torch import nn
import loss
# from complexPyTorch.complexLayers import (
#     #     ComplexConv2d as Conv2d,
#     #     ComplexReLU as ReLU,
#     ComplexMaxPool2d as MaxPool2d,
# )
from torch.nn import Conv2d, LeakyReLU, MaxPool2d
from torchvision.transforms import CenterCrop


class LitUnet(pl.LightningModule):
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
        b1,
        b2,
        **kwargs,
    ):
        super().__init__()
        self.out_size = [stft_height, stft_width]
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.betas = (b1, b2)

        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = loss.MSLELoss()
        # self.loss_fn = loss.ComplexMSLELoss()
        self.loss_fn = loss.MagnitudeMSELoss()

        scale_factor = (2, 1)
        self.down = MaxPool2d(scale_factor, ceil_mode=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.down_blocks = nn.ModuleList([
            self.block(in_channels, 128, 9, "down"),
            self.block(128, 128, 5, "down"),
            self.block(128, 512, 3, "down"),
            self.block(512, 512, 3, "down"),
        ])
        self.bottom = self.block(512, 512, 3, "bottom")
        self.up_blocks = nn.ModuleList([
            self.block(512, 512, 3, "up"),
            self.block(512, 128, 3, "up"),
            self.block(128, 128, 3, "up"),
            self.block(128, 8, 3, "up"),
        ])
        self.out = Conv2d(8, out_channels, kernel_size=3)
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
        parser.add_argument("--b1", type=float, default=0.9)
        parser.add_argument("--b2", type=float, default=0.999)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--in_channels", type=int, default=2)
        parser.add_argument("--out_channels", type=int, default=2)
        return parent_parser

    def block(self, in_channels, out_channels, kernel_height, direction):
        in_channels = 2*in_channels if direction == "up" else in_channels
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_height, 1),
            padding="same",
            dilation=2 if direction == "down" else 1,
        )
        if direction == "down":
            post = nn.LeakyReLU(0.2)
        elif direction == "bottom":
            post = nn.Sequential(nn.Dropout2d(), nn.LeakyReLU(0.2))
        else:
            post = nn.Sequential(nn.Dropout2d(), nn.ReLU())
        return nn.Sequential(conv, post)

    def crop_width_height(self, x, shape_to_match):
        return CenterCrop(shape_to_match[-2:])(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device)

        phase = torch.angle(torch.view_as_complex(x))
        x = torch.abs(torch.view_as_complex(x))
        # x = torch.cat([magnitude, phase], dim=1)

        x = x.unsqueeze(1)

        x_in = x.clone()
        # Convert real/imag axes to channels
        # x = x.permute(0, 3, 1, 2)

        # x = self.crop_width_height(x, [self.out_size[0]-1, self.out_size[1]])
        # x = torch.cat(x_subs, dim=1)

        skip = []

        for block in self.down_blocks:
            x = block(x)
            skip.append(x.clone())
            x = self.down(x)
        x = self.bottom(x)

        for skip_connection, block in zip(skip[::-1], self.up_blocks):
            x = self.up(x)
            skip_connection = self.crop_width_height(
                skip_connection, x.shape)
            x = torch.cat((x, skip_connection), dim=1)
            x = block(x)

        # Merge input with output
        x = torch.cat((x_in, x), dim=1)

        x = self.out(x)

        # x_subs = torch.split(x, round(x.shape[1]/4), dim=1)
        # x = torch.cat(x_subs, dim=2)

        x = self.crop_width_height(x, self.out_size)

        # Convert channels to real/imag axes
        # x = x.permute(0, 2, 3, 1)
        x = x.squeeze(1)

        x = torch.polar(x, phase)
        x = torch.view_as_real(x)

        return x

    def _step(self, batch, batch_idx, step_name):
        """Generic code to run for each step in train/val/test"""
        (x, _), (y, _) = batch
        pred = self(x)

        # pred_top = torch.view_as_real(
        #     self.crop_width_height(
        #         torch.view_as_complex(pred),
        #         [round(self.out_size[0]*0.75), self.out_size[1]]
        #     )
        # )
        # y_top = torch.view_as_real(
        #     self.crop_width_height(
        #         torch.view_as_complex(y),
        #         [round(self.out_size[0]*0.75), self.out_size[1]]
        #     )
        # )
        # loss = self.loss_fn(pred_top, y_top)
        loss = self.loss_fn(pred, y)
        # loss = torch.sqrt(loss)
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
            return torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        elif self.optim == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            raise Exception(f"Optimizer '{self.optim}' not recognized")
