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
        self.g_h2z, self.g_z2h, self.d_h, self.d_z = get_networks()
        self.loss = Losses()

        batch_size = config.training.batch_size
        self.target_real = torch.autograd.Variable(torch.ones(batch_size, device=self.device), requires_grad=False)
        self.target_fake = torch.autograd.Variable(torch.zeros(batch_size, device=self.device), requires_grad=False)

        self.automatic_optimization = False

    @property
    def lr(self):
        return self.optimizers().param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        real_h, real_z = batch['horse'], batch['zebra']
        g_opt, d_opt, = self.optimizers()
        # augment_images = DiffAugment(images, policy='color,translation,cutout')

        # ----------------- #
        #   Fit generator   #
        # ----------------- #
        # Identity loss
        same_h, same_z = self.g_z2h(real_h), self.g_h2z(real_z)
        loss_identity_h = self.loss.criterion_identity(same_h, real_h)
        loss_identity_z = self.loss.criterion_identity(same_z, real_z)

        # GAN loss
        fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)
        pred_fake_h, pred_fake_z = self.d_h(fake_h), self.d_z(fake_z)
        loss_gan_z2h = self.loss.criterion_GAN(pred_fake_h, self.traget_real)
        loss_gan_h2z = self.loss.criterion_GAN(pred_fake_z, self.traget_real)

        # Cycle loss
        recovered_h, recovered_z = self.g_z2h(fake_z), self.g_h2z(fake_h)
        loss_cycle_hzh = self.loss.criterion_cycle(recovered_h, real_h)
        loss_cycle_zhz = self.loss.criterion_cycle(recovered_z, real_z)

        # Total loss
        loss_g = loss_identity_h + loss_identity_z + loss_gan_z2h + loss_gan_h2z + loss_cycle_hzh + loss_cycle_zhz

        # Step
        g_opt.zero_grad()
        loss_g.backward()
        g_opt.step()

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
        optimizer_g = torch.optim.Adam(list(self.g_h2z.parameters()) + list(self.g_z2h.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(list(self.d_h.parameters()) + list(self.d_z.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        return optimizer_g, optimizer_d


