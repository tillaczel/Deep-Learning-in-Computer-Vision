from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn

from project_3.src.model import get_networks
from project_3.src.loss import Losses
from project_3.src.DiffAugment_pytorch import DiffAugment


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.G_A2B, self.G_B2A, self.D_A, self.D_B = get_networks()
        self.losses = Losses()

        self.automatic_optimization = False

    @property
    def lr(self):
        return self.optimizers().param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print(batch)

        g_opt, d_opt, = self.optimizers()
        augment_images = DiffAugment(images, policy='color,translation,cutout')

        self.log('loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        return {'loss': loss}

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs: list):
        pass

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(list(self.G_A2B.parameters())+list(self.G_B2A.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(list(self.D_A.parameters())+list(self.D_B.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        return optimizer_g, optimizer_d


