import pytorch_lightning as pl
import torch
from torch import nn

class LitModel(pl.LightningModule):
    def __init__(self, in_features:int, out_features:int, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.MSELoss(reduction="sum")
        mid_features = 128
        self.layers = nn.Sequential(
            # nn.Linear(in_features, in_features),
            # nn.Linear(in_features, out_features),
            nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1),
            nn.Conv2d(mid_features, mid_features, kernel_size=3, padding=1),
            nn.Conv2d(mid_features, mid_features, kernel_size=3, padding=1),
            nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1),
        )
        
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[-3:])
        print(f"forward:\t{x.shape}")
        x = self.layers(x)
        return x
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Returns loss from single batch"""
        (x, x_sr), (y, y_sr) = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        """Logs validation loss"""
        (x, x_sr), (y, y_sr) = batch
        pred = self(x)
        print(f"val pred:\t{pred.shape}")
        print(f"val y:\t\t{y.shape}")
        loss = self.loss_fn(pred, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        """Logs test loss"""
        (x, x_sr), (y, y_sr) = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    