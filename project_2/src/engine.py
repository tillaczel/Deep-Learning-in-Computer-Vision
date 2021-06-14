from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics

from .model import Model
from .plot_results import *


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig, main_metric: str="acc"):
        super().__init__()
        self.config = config
        self.model = Model(n_channels=config.model.in_dim, n_classes=config.model.out_dim)
        self.loss_func = nn.BCEWithLogitsLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.train_sensitivity = torchmetrics.Recall()
        self.val_sensitivity = torchmetrics.Recall()
        self.train_specificity = torchmetrics.Specificity()
        self.val_specificity = torchmetrics.Specificity()


        self.metrics = ["acc", "f1", "sensitivity", "specificity"]
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
        seg_hat = self.model(images)
        loss = self.loss_func(seg_hat, labels)

        self.log('loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        
        
        ### TESTING PURPOSES ONLY
        
        dataset = self.trainer.val_dataloaders[0].dataset
        print(dataset)
        images, segmentations = dataset[0]
        #images, segmentations = torch.unsqueeze(images,0), torch.unsqueeze(segmentations,0)
        images, segmentations = map(torch.unsqueeze, [(images,0), (segmentations,0)])
        
        
        #torch.unsqueeze(images,0), torch.unsqueeze(segmentations,0)
        print(images.shape)
        print(segmentations.shape)
        
        preds = self.model(images)  # Do a forward pass of validation data to get predictions
        plot_predictions(dataset, preds)
        
        return {'loss': loss}

    def training_epoch_end(self, outputs: list):
        # TESTING PURPOSES ONLY
        dataset = self.trainer.val_dataloaders[0].dataset
        print(dataset)
        images, segmentations = dataset[0]
        
        preds = self.model(images)  # Do a forward pass of validation data to get predictions
        plot_predictions(dataset, preds)
        
        pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        seg_hat = self.model(images)
        loss = self.loss_func(seg_hat, labels)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: list):
        #dataset = self.trainer.val_dataloaders[0].dataset
        #images = dataset[0]
        #labels = dataset[1]
        #preds = self.model(dataset)  # Do a forward pass of validation data to get predictions
        #plot_predictions(dataset, preds)
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
