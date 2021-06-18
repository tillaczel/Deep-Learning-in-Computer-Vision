import os
import wandb
from omegaconf import DictConfig, OmegaConf

from project_3.src.data import get_dataloaders
from project_3.src.engine import EngineModule
from project_3.src.trainer import get_trainer


def run_training(cfg: DictConfig):
    cfg_file = os.path.join(wandb.run.dir, 'train_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file, base_path=wandb.run.dir)  # this will force sync it

    wandb.save('*.ckpt')  # should keep it up to date
    engine = EngineModule(cfg)
    trainer = get_trainer(cfg, engine)

    train_loader, valid_loader = \
        get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, cfg.data.url,
                        cfg.data.path)
    trainer.fit(engine, train_dataloader=train_loader, val_dataloaders=valid_loader)
