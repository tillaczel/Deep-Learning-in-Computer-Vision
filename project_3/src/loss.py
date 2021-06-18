import pytorch_lightning as pl
import torch


class Losses(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
