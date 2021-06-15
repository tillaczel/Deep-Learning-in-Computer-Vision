from collections import Iterable

from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn
from project_2.src.model import Model
from project_2.src.plot_results import plot_predictions
from project_2.src.metrics import Metrics


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig,
                 main_metrics: Iterable = ("sensitivity", "specificity", "iou", "dice", "acc")):
        super().__init__()
        self.config = config
        self.model = Model(n_channels=config.model.in_dim, n_classes=config.model.out_dim, dropout_rate=config.model.dropout_rate)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.metrics = Metrics(main_metrics)

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
        images, labels = batch
        seg_hat = self.model(images)
        loss = self.loss_func(seg_hat, labels.type(torch.float32))

        self.log('loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        probs = torch.sigmoid(seg_hat)
        for metric_name in self.metrics.metrics:
            self.update_and_log_metric(metric_name, probs, labels.type(torch.long), mode='train')

        return {'loss': loss}

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        seg_hat = self.model(images)
        loss = self.loss_func(seg_hat, labels.type(torch.float32))

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        probs = torch.sigmoid(seg_hat)
        for metric_name in self.metrics.metrics:
            self.update_and_log_metric(metric_name, probs, labels.type(torch.long), mode='val')

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: list):
        plot_predictions(self.trainer.val_dataloaders[0].dataset, self.model, self.device,
                         current_epoch=self.current_epoch, mode='single')
        if self.model.dropout_rate > 0:
            plot_predictions(self.trainer.val_dataloaders[0].dataset, self.model, self.device,
                             current_epoch=self.current_epoch, mode='mc_dropout')

        pass

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.training.optimizer, self.parameters())
        scheduler_config = self.config.training.scheduler
        if scheduler_config is not None:
            scheduler = get_scheduler(scheduler_config, optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer


def get_optimizer(optim_config: DictConfig, params):
    name = optim_config.name
    lr = optim_config.lr

    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f'{name} not in optimizers')


def get_scheduler(scheduler_config, optimizer):
    name = scheduler_config.name
    monitor = scheduler_config.monitor

    if name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=scheduler_config.mode,
                                                               patience=scheduler_config.patience,
                                                               factor=scheduler_config.factor,
                                                               min_lr=scheduler_config.min_lr)
        return dict(scheduler=scheduler, monitor=monitor)
    else:
        raise ValueError(f'{name} not in schedulers')
