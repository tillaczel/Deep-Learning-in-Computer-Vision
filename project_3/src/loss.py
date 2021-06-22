import pytorch_lightning as pl
import torch


class Losses(pl.LightningModule):
    def __init__(self, d_loss, g_loss):
        super().__init__()
        if d_loss == 'mse':
            self.criterion_GAN = torch.nn.MSELoss()
        elif d_loss == 'bce':
            self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError('wrong d_loss')

        if g_loss == 'l1':
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_identity = torch.nn.L1Loss()
        elif g_loss == 'l2':
            self.criterion_cycle = torch.nn.MSELoss()
            self.criterion_identity = torch.nn.MSELoss()
        else:
            raise ValueError('wrong g_loss')

