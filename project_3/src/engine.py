from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn
import itertools

from project_3.src.model import Model
from project_3.src.plot_results import plot_predictions


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig, main_metrics=None):
        super().__init__()
        self.config = config
        self.G_A2B, self.G_B2A, self.D_A, self.D_B = None, None, None, None
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.training.loss.pos_weight]))

        self.automatic_optimization = False

    @property
    def lr(self):
        return self.optimizers().param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)

    def update_and_log_metric(self, metric_name, probs, labels, mode='train'):
        metric = getattr(self.metrics, f"{mode}_{metric_name}")
        metric(probs, labels)
        self.log(f"{mode}_{metric_name}", metric,
                 on_step=False,
                 prog_bar=(metric_name in self.metrics.main_metrics),
                 on_epoch=True, logger=True)

    def training_step(self, batch, batch_idx):
        g_opt, d_a_opt, d_b_opt = self.optimizers()
        images, labels = batch
        seg_hat = self.model(images)
        loss = self.loss_func(seg_hat, labels.type(torch.float32))

        self.log('loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        probs = torch.sigmoid(seg_hat)
        for metric_name in self.metrics.metrics:
            self.update_and_log_metric(metric_name, probs, (labels >= 0.5).int(), mode='train')

        return {'loss': loss}

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        seg_hat = self.model(images)
        loss = self.loss_func(torch.moveaxis(seg_hat, 1, -1),
                              torch.moveaxis(labels.type(torch.float32), 1, -1))

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        probs = torch.sigmoid(seg_hat)
        for metric_name in self.metrics.metrics:
            self.update_and_log_metric(metric_name, probs, (labels >= 0.5).int(), mode='val')

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: list):
        plot_predictions(self.trainer.val_dataloaders[0].dataset, self.model, self.device,
                         current_epoch=self.current_epoch)
        pass

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.netG_B2A.parameters()),
                                       lr=self.config.training.lr, betas=(0.5, 0.999))
        optimizer_d_a = torch.optim.Adam(self.D_A.parameters(), lr=self.config.training.lr, betas=(0.5, 0.999))
        optimizer_d_b = torch.optim.Adam(self.D_B.parameters(), lr=self.config.training.lr, betas=(0.5, 0.999))
        return optimizer_g, optimizer_d_a, optimizer_d_b


