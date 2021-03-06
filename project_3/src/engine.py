from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models, transforms

from project_3.src.fid import FrechetInceptionDistance
from project_3.src.model import get_networks
from project_3.src.loss import Losses
from project_3.src.DiffAugment_pytorch import DiffAugment
from project_3.src.plot_results import make_plots
from project_3.src.utils import ImagePool

from pytorch_fid.inception import InceptionV3


class EngineModule(pl.LightningModule):

    def __init__(self, config: DictConfig, test_dataset_horse, test_dataset_zebra):
        super().__init__()
        self.config = config
        self.g_h2z, self.g_z2h, self.d_h, self.d_z = get_networks(config.model.f, config.model.blocks)
        self.loss = Losses(config.training.d_loss, config.training.g_loss)

        batch_size = config.training.batch_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target_real = torch.autograd.Variable(torch.ones((batch_size, 196), device=device), requires_grad=False)
        self.target_fake = torch.autograd.Variable(torch.zeros((batch_size, 196), device=device), requires_grad=False)
        self.target_real_val = torch.ones((20, 196), device=device)
        self.target_fake_val = torch.zeros((20, 196), device=device)

        self.test_dataset_horse = test_dataset_horse
        self.test_dataset_zebra = test_dataset_zebra

        self.fake_pool_H = ImagePool(pool_sz=300)
        self.fake_pool_Z = ImagePool(pool_sz=300)

        # self.inception =  models.inception_v3(pretrained=True)
        # # TODO cut layer
        # self.inception.fc = nn.Identity() # not sure if it works

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

        self.inception = InceptionV3([block_idx])

        self.fid_identity_h = FrechetInceptionDistance(self.inception)
        self.fid_identity_z = FrechetInceptionDistance(self.inception)
        self.fid_cycle_h = FrechetInceptionDistance(self.inception)
        self.fid_cycle_z = FrechetInceptionDistance(self.inception)
        self.fid_h2z = FrechetInceptionDistance(self.inception)
        self.fid_z2h = FrechetInceptionDistance(self.inception)

        self.warmup_epochs = config.training.warmup_epochs
        self.weight_identity = config.training.weight_identity
        self.weight_cycle = config.training.weight_cycle
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def get_loss_weights(self):
        # warmup gan weights
        if self.current_epoch < self.warmup_epochs:
            d = self.current_epoch / self.warmup_epochs
            g_gan = d
        else:
            d = 1
            g_gan = 1

        return d, self.weight_identity, g_gan, self.weight_cycle

    def training_step(self, batch, batch_idx):
        loss_weight_d, loss_weight_g_identity, loss_weight_g_gan, loss_weight_g_cycle = self.get_loss_weights()

        real_h, real_z = batch['horse'], batch['zebra']
        g_opt, d_opt, = self.optimizers()
        # augment_images = DiffAugment(images, policy='color,translation,cutout')

        # TODO: add loss weights
        # TODO: normalizatoin
        # --------------------- #
        #   Fit discriminator   #
        # --------------------- #

        # Real loss

        if self.config.training.augment:
            pred_h_real, pred_z_real = self.d_h(DiffAugment(real_h, policy='color,translation,cutout')), self.d_z(
                DiffAugment(real_z, policy='color,translation,cutout'))
        else:
            pred_h_real, pred_z_real = self.d_h(real_h), self.d_z(real_z)

        loss_h_real = self.loss.criterion_GAN(pred_h_real, self.target_real)
        loss_z_real = self.loss.criterion_GAN(pred_z_real, self.target_real)
        del pred_h_real, pred_z_real


        # Fake loss
        fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)

        # Sample from 50 previous generated images
        fake_h = self.fake_pool_H.push_and_pop(fake_h, self.device)
        fake_z = self.fake_pool_Z.push_and_pop(fake_z, self.device)

        if self.config.training.augment:
            pred_h_fake, pred_z_fake = self.d_h(
            DiffAugment(fake_h.detach(), policy='color,translation,cutout')), self.d_z(
            DiffAugment(fake_z.detach(), policy='color,translation,cutout'))
        else:
            pred_h_fake, pred_z_fake = self.d_h(fake_h.detach()), self.d_z(fake_z.detach())
        loss_h_fake = self.loss.criterion_GAN(pred_h_fake, self.target_fake)
        loss_z_fake = self.loss.criterion_GAN(pred_z_fake, self.target_fake)
        del fake_h, fake_z, pred_h_fake, pred_z_fake

        # Total loss
        loss_d_unweighted = loss_h_real + loss_z_real + loss_h_fake + loss_z_fake
        loss_d = loss_weight_d * loss_d_unweighted

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
        if self.config.training.augment:
            pred_fake_h, pred_fake_z = self.d_h(DiffAugment(fake_h, policy='color,translation,cutout')), self.d_z(DiffAugment(fake_z, policy='color,translation,cutout'))
        else:
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
        loss_g = loss_weight_g_identity * (loss_identity_h + loss_identity_z) + loss_weight_g_gan * (loss_gan_z2h + loss_gan_h2z) + loss_weight_g_cycle * (loss_cycle_hzh + loss_cycle_zhz)
        loss_g_unweighted = loss_identity_h + loss_identity_z + loss_gan_z2h + loss_gan_h2z + loss_cycle_hzh + loss_cycle_zhz

        # Step generator
        g_opt.zero_grad()
        loss_g.backward()
        g_opt.step()

        # Log discriminator
        self.log('loss_d_h_real', loss_h_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_d_z_real', loss_z_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_d_h_fake', loss_h_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_d_z_fake', loss_z_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_d_real', loss_h_real+loss_z_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_d_fake', loss_h_fake+loss_z_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_d', loss_d, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_d_unweighted', loss_d_unweighted, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log generator
        self.log('loss_g_gan', loss_gan_z2h+loss_gan_h2z, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_g_cycle', loss_cycle_hzh+loss_cycle_zhz, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_g', loss_g, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_g_unweighted', loss_g_unweighted, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        self.log('loss_identity_h', loss_identity_h, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_identity_z', loss_identity_z, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_identity', loss_identity_h+loss_identity_z, on_step=False, on_epoch=False, prog_bar=True, logger=True)
        self.log('loss_gan_z2h', loss_gan_z2h, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_gan_h2z', loss_gan_h2z, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_cycle_hzh', loss_cycle_hzh, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_cycle_zhz', loss_cycle_zhz, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log('loss_h_real', loss_h_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_z_real', loss_z_real, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_h_fake', loss_h_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_z_fake', loss_z_fake, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_sum', loss_g_unweighted + loss_d_unweighted, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss_g': loss_g, 'loss_d': loss_d}

    def training_epoch_end(self, outputs: list):
        pass

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            real_h, real_z = batch['horse'], batch['zebra']

            if real_h.size()[-1] == 0:
                return
            # TODO: normalizatoin
            # --------------------- #
            #   Fit discriminator   #
            # --------------------- #

            # Real loss
            pred_h_real, pred_z_real = self.d_h(real_h), self.d_z(real_z)
            loss_h_real = self.loss.criterion_GAN(pred_h_real, self.target_real_val)
            loss_z_real = self.loss.criterion_GAN(pred_z_real, self.target_real_val)
            del pred_h_real, pred_z_real

            # Fake loss
            fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)
            pred_h_fake, pred_z_fake = self.d_h(fake_h), self.d_z(fake_z)
            loss_h_fake = self.loss.criterion_GAN(pred_h_fake, self.target_fake_val)
            loss_z_fake = self.loss.criterion_GAN(pred_z_fake, self.target_fake_val)
            del fake_h, fake_z, pred_h_fake, pred_z_fake

            # Total loss
            loss_d = loss_h_real + loss_z_real + loss_h_fake + loss_z_fake

            # --------------------- #
            #     Fit generator     #
            # --------------------- #
            # Identity loss
            same_h, same_z = self.g_z2h(real_h), self.g_h2z(real_z)
            loss_identity_h = self.loss.criterion_identity(same_h, real_h)
            loss_identity_z = self.loss.criterion_identity(same_z, real_z)

            # TODO: FID ?
            self.fid_identity_h.update(same_h, real_h)
            self.fid_identity_z.update(same_z, real_z)
            del same_h, same_z


            # GAN loss
            fake_h, fake_z = self.g_z2h(real_z), self.g_h2z(real_h)
            pred_fake_h, pred_fake_z = self.d_h(fake_h), self.d_z(fake_z)
            loss_gan_z2h = self.loss.criterion_GAN(pred_fake_h, self.target_real_val)
            loss_gan_h2z = self.loss.criterion_GAN(pred_fake_z, self.target_real_val)
            # TODO: should this be h2z vs z or h2z vs h ?
            self.fid_h2z.update(fake_z, real_z)
            self.fid_z2h.update(fake_h, real_h)
            del pred_fake_h, pred_fake_z

            # Cycle loss
            recovered_h, recovered_z = self.g_z2h(fake_z), self.g_h2z(fake_h)
            loss_cycle_hzh = self.loss.criterion_cycle(recovered_h, real_h)
            loss_cycle_zhz = self.loss.criterion_cycle(recovered_z, real_z)
            self.fid_cycle_h.update(recovered_h, real_h)
            self.fid_cycle_z.update(recovered_z, real_z)
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
        self.log('val_fid_identity_h', self.fid_identity_h.compute(), on_epoch=True, prog_bar=False)
        self.log('val_fid_identity_z', self.fid_identity_z.compute(), on_epoch=True, prog_bar=False)
        self.log('val_fid_cycle_h', self.fid_cycle_h.compute(), on_epoch=True, prog_bar=False)
        self.log('val_fid_cycle_z', self.fid_cycle_z.compute(), on_epoch=True, prog_bar=False)
        self.log('val_fid_h2z', self.fid_h2z.compute(), on_epoch=True, prog_bar=False)
        self.log('val_fid_z2h', self.fid_z2h.compute(), on_epoch=True, prog_bar=False)

        # TODO: this should be redundant
        self.fid_identity_h.reset()
        self.fid_identity_z.reset()
        self.fid_cycle_h.reset()
        self.fid_cycle_z.reset()
        self.fid_h2z.reset()
        self.fid_z2h.reset()

        self.visualize()


    def visualize(self, n=10):
        make_plots(self.test_dataset_horse, self.g_h2z, self.g_z2h, self.device, n=n, current_epoch=self.current_epoch,
                   suffix='_h2z')
        make_plots(self.test_dataset_zebra, self.g_z2h, self.g_h2z, self.device, n=n, current_epoch=self.current_epoch,
                   suffix='_z2h')

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(list(self.g_h2z.parameters()) + list(self.g_z2h.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(list(self.d_h.parameters()) + list(self.d_z.parameters()),
                                       lr=self.config.training.optimizer.lr, betas=(0.5, 0.999))
        return optimizer_g, optimizer_d


