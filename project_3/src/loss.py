import pytorch_lightning as pl
import torch


class Losses(pl.LightningModule):
    def __init__(self, d_loss):
        super().__init__()
        if d_loss == 'mse':
            self.criterion_GAN = torch.nn.MSELoss()
        elif d_loss == 'bce':
            self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError('wrong d_loss')
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
