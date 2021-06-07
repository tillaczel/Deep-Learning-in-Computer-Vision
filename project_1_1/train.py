import sys
import os

sys.path.append('git_repo')

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from project_1_1.src.data import get_data
from project_1_1.src.engine import EngineModule
from project_1_1.src.data import download_data

wandb.init(project='p1', entity='dlcv')


@hydra.main(config_path='config', config_name="default")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_file = os.path.join(wandb.run.dir, 'config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)  # this will force sync it

    download_data(cfg_file.data.path)
    train_dataloader, test_dataloader = get_data(cfg.data.size, cfg.data.batch_size, base_path=cfg_file.data.path)
    engine = EngineModule(cfg)

    wandb.save('*.ckpt')  # should keep it up to date

    trainer = get_trainer(cfg, engine)

    trainer.fit(engine, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)

    # TODO: visualizations


if __name__ == '__main__':
    run_training()


def get_trainer(cfg, engine):
    callbacks = list()
    callbacks.append(pl.callbacks.EarlyStopping(patience=cfg.training.early_stopping.stopping_patience,
                                                monitor=cfg.training.early_stopping.monitor))
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=wandb.run.dir,
                                                  monitor=cfg.training.model_checkpoint.monitor,
                                                  filename='model',
                                                  verbose=True,
                                                  period=1))

    gpus = 0
    if torch.cuda.is_available():
        gpus = -1

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)

    # TODO: we can use this to put configuration into nice table in wandb
    # logger.log_hyperparams(hyperparams)

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir="training/logs",
                         max_epochs=cfg.training.max_epochs, gpus=gpus)
    return trainer
