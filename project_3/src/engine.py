from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn

from project_3.src.model import get_networks
from project_3.src.loss import Losses
from project_3.src.DiffAugment_pytorch import DiffAugment
from project_3.src.plot_results import make_plots


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig, test_dataset_horse, test_dataset_zebra):
        super().__init__()
        self.config = config
        self.g_h2z, self.g_z2h, self.d_h, self.d_z = get_networks()
        self.loss = Losses()

        batch_size = config.training.batch_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target_real = torch.autograd.Variable(torch.ones((batch_size, 1), device=device), requires_grad=False)
        self.target_fake = torch.autograd.Variable(torch.zeros((batch_size, 1), device=device), requires_grad=False)
        self.test_dataset_horse = test_dataset_horse
        self.test_dataset_zebra = test_dataset_zebra

        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        real_h, real_z = batch['horse'], batch['zebra']
        g_opt, d_opt, = self.optimizers()
        # augment_images = DiffAugment(images, policy='color,translation,cutout')

        # TODO: add loss weights
        # TODO: normalizatoin
        # --------------------- #
        #   Fit discriminator   #
        # --------------------- #

        # Real loss
        pred_h_real, pred_z_real = self.d_h(real_h), self.d_z(real_z)
        loss_h_real = self.loss.criterion_GAN(pred_h_real, self.target_real)
        loss_z_real = self.loss.criterion_GAN(pred_z_real, self.target_real)
        del pred_h_real, pred_z_real

        # Fake loss
        fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)
        pred_h_fake, pred_z_fake = self.d_h(fake_h.detach()), self.d_z(fake_z.detach())
        loss_h_fake = self.loss.criterion_GAN(pred_h_fake, self.target_fake)
        loss_z_fake = self.loss.criterion_GAN(pred_z_fake, self.target_fake)
        del fake_h, fake_z, pred_h_fake, pred_z_fake

        # Total loss
        loss_d = loss_h_real+loss_z_real+loss_h_fake+loss_z_fake

        # Step discriminator
        d_opt.zero_grad()
        loss_d.backward()
        d_opt.step()

        # --------------------- #
        #     Fit generator     #
        # --------------------- #
        # Identity loss
        same_h, same_z = self.g_z2h(real_h), self.g_h2z(real_z)
        loss_identity_h = self.loss.criterion_identity(same_h, real_h)
        loss_identity_z = self.loss.criterion_identity(same_z, real_z)
        del same_h, same_z

        # GAN loss
        fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)
        pred_fake_h, pred_fake_z = self.d_h(fake_h), self.d_z(fake_z)
        loss_gan_z2h = self.loss.criterion_GAN(pred_fake_h, self.target_real)
        loss_gan_h2z = self.loss.criterion_GAN(pred_fake_z, self.target_real)
        del pred_fake_h, pred_fake_z

        # Cycle loss
        recovered_h, recovered_z = self.g_z2h(fake_z), self.g_h2z(fake_h)
        loss_cycle_hzh = self.loss.criterion_cycle(recovered_h, real_h)
        loss_cycle_zhz = self.loss.criterion_cycle(recovered_z, real_z)
        del fake_h, fake_z, recovered_h, recovered_z

        # Total loss
        loss_g = loss_identity_h + loss_identity_z + loss_gan_z2h + loss_gan_h2z + loss_cycle_hzh + loss_cycle_zhz

        # Step generator
        g_opt.zero_grad()
        loss_g.backward()
        g_opt.step()


        self.log('loss_g', loss_g, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_d', loss_d, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # for tracking general progress
        self.log('loss_sum', loss_d + loss_g, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log('loss_identity_h', loss_identity_h, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_identity_z', loss_identity_z, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_gan_z2h', loss_gan_z2h, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_gan_h2z', loss_gan_h2z, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_cycle_hzh', loss_cycle_hzh, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_cycle_zhz', loss_cycle_zhz, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log('loss_h_real', loss_h_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_z_real', loss_z_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_h_fake', loss_h_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_z_fake', loss_z_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return {'loss_g': loss_g, 'loss_d': loss_d}

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            real_h, real_z = batch['horse'], batch['zebra']
            print(real_h.shape, real_h.size())

            if len(real_h.size()) == 0:
                return

            g_opt, d_opt, = self.optimizers()
            # TODO: normalizatoin
            # --------------------- #
            #   Fit discriminator   #
            # --------------------- #

            # Real loss
            pred_h_real, pred_z_real = self.d_h(real_h), self.d_z(real_z)
            loss_h_real = self.loss.criterion_GAN(pred_h_real, self.target_real)
            loss_z_real = self.loss.criterion_GAN(pred_z_real, self.target_real)
            del pred_h_real, pred_z_real

            # Fake loss
            fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)
            pred_h_fake, pred_z_fake = self.d_h(fake_h), self.d_z(fake_z)
            loss_h_fake = self.loss.criterion_GAN(pred_h_fake, self.target_fake)
            loss_z_fake = self.loss.criterion_GAN(pred_z_fake, self.target_fake)
            del fake_h, fake_z, pred_h_fake, pred_z_fake

            # Total loss
            loss_d = loss_h_real+loss_z_real+loss_h_fake+loss_z_fake

            # --------------------- #
            #     Fit generator     #
            # --------------------- #
            # Identity loss
            same_h, same_z = self.g_z2h(real_h), self.g_h2z(real_z)
            loss_identity_h = self.loss.criterion_identity(same_h, real_h)
            loss_identity_z = self.loss.criterion_identity(same_z, real_z)
            del same_h, same_z

            # GAN loss
            fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)
            pred_fake_h, pred_fake_z = self.d_h(fake_h), self.d_z(fake_z)
            loss_gan_z2h = self.loss.criterion_GAN(pred_fake_h, self.target_real)
            loss_gan_h2z = self.loss.criterion_GAN(pred_fake_z, self.target_real)
            del pred_fake_h, pred_fake_z

            # Cycle loss
            recovered_h, recovered_z = self.g_z2h(fake_z), self.g_h2z(fake_h)
            loss_cycle_hzh = self.loss.criterion_cycle(recovered_h, real_h)
            loss_cycle_zhz = self.loss.criterion_cycle(recovered_z, real_z)
            del fake_h, fake_z, recovered_h, recovered_z

            # Total loss
            loss_g = loss_identity_h + loss_identity_z + loss_gan_z2h + loss_gan_h2z + loss_cycle_hzh + loss_cycle_zhz

            self.log('val_loss_g', loss_g, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_d', loss_d, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # for tracking general progress
            self.log('val_loss_sum', loss_d + loss_g, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            self.log('val_loss_identity_h', loss_identity_h, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_identity_z', loss_identity_z, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_gan_z2h', loss_gan_z2h, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_gan_h2z', loss_gan_h2z, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_cycle_hzh', loss_cycle_hzh, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_cycle_zhz', loss_cycle_zhz, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            self.log('val_loss_h_real', loss_h_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_z_real', loss_z_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_h_fake', loss_h_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_z_fake', loss_z_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def validation_epoch_end(self, outputs: list):
        make_plots(self.test_dataset_horse, self.g_h2z, self.g_z2h, self.device, n=4, current_epoch=self.current_epoch, suffix='_h2z')
        make_plots(self.test_dataset_zebra, self.g_z2h, self.g_h2z, self.device, n=4, current_epoch=self.current_epoch, suffix='_z2h')

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(list(self.g_h2z.parameters()) + list(self.g_z2h.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(list(self.d_h.parameters()) + list(self.d_z.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        return optimizer_g, optimizer_d


