from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics

from .model import Model


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig, n_classes: int=11, main_metric: str="f1"):
        super().__init__()
        self.config = config
        self.model = Model(pretrained=config.model.pretrained, in_dim=config.model.in_dim, out_dim=config.model.out_dim)
        self.loss_func = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(multiclass=True, num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(multiclass=True, num_classes=n_classes)

        self.train_f1 = torchmetrics.F1(multiclass=True, num_classes=n_classes)
        self.val_f1 = torchmetrics.F1(multiclass=True, num_classes=n_classes)


        self.metrics = ["acc", "f1"]
        self.main_metric = main_metric

    @property
    def lr(self):
        return self.optimizers().param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)

    def update_and_log_metric(self, metric_name, probs, labels, mode='train'):
        metric = getattr(self, f"{mode}_{metric_name}")
        metric(probs, labels)
        self.log(f"{mode}_{metric_name}", metric,
                 on_step=False,
                 prog_bar=(metric_name == self.main_metric),
                 on_epoch=True, logger=True)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        pred = self.model(images).squeeze()  # [Bx1] -> [B]
        loss = self.loss_func(pred, labels.type(torch.long))
        self.log('loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        probs = nn.functional.softmax(pred, dim=-1)

        for metric_name in self.metrics:
            self.update_and_log_metric(metric_name, probs, labels, mode='train')

        return {'loss': loss}

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        pred = self.model(images).squeeze()  # [Bx1] -> [B]
        loss = self.loss_func(pred, labels.type(torch.long))

        probs = nn.functional.softmax(pred, dim=-1)
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        for metric_name in self.metrics:
            self.update_and_log_metric(metric_name, probs, labels, mode='val')


        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: list):
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
