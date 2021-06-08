from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics

from .model import Model

class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = Model(pretrained=config.model.pretrained, in_dim=config.model.in_dim, out_dim=config.model.out_dim)
        self.loss_func = nn.BCEWithLogitsLoss()
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        
        self.train_F1 = torchmetrics.F1(num_classes=2, multiclass=True) # Not sure if arguments are right
        self.valid_F1 = torchmetrics.F1(num_classes=2, multiclass=True)
        

    @property
    def lr(self):
        return self.optimizers().param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        pred = self.model(images).squeeze()  # [Bx1] -> [B]
        loss = self.loss_func(pred, labels.type(torch.float32))
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
     
        train_acc = self.train_acc((pred>0.5).float().to(device='cuda:0'), labels.type(torch.IntTensor).to(device='cuda:0'))
        train_F1 = self.train_F1((pred>0.5).float().to(device='cuda:0'), labels.type(torch.IntTensor).to(device='cuda:0'))
        self.log('train_acc', train_acc, on_step=True, on_epoch=False)
        self.log('train_F1', train_F1, on_step=True, on_epoch=False)
        
        return {'loss': loss, 'train_acc': train_acc, 'train_F1': train_F1}

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        pred = self.model(images).squeeze()  # [Bx1] -> [B]
        loss = self.loss_func(pred, labels.type(torch.float32))
        
        valid_acc = self.valid_acc((pred>0.5).float().to(device='cuda:0'), labels.type(torch.IntTensor).to(device='cuda:0'))
        valid_F1 = self.valid_F1((pred>0.5).float().to(device='cuda:0'), labels.type(torch.IntTensor).to(device='cuda:0'))
        
        self.log('valid_acc', valid_acc, on_step=True, on_epoch=True)
        self.log('valid_F1', valid_F1, on_step=True, on_epoch=True)
        
        return {'val_loss': loss, 'valid_acc': valid_acc, 'valid_F1': valid_F1}

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
