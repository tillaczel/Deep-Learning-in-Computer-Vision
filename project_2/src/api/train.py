import os
import wandb
from omegaconf import DictConfig, OmegaConf

from project_2.src.data import get_dataloaders
from project_2.src.engine import EngineModule
from project_2.src.trainer import get_trainer


def run_training(cfg: DictConfig):
    cfg_file = os.path.join(wandb.run.dir, 'train_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)  # this will force sync it

    engine = EngineModule(cfg)
    wandb.save('*.ckpt')  # should keep it up to date
    trainer = get_trainer(cfg, engine)

    if cfg.model.ensemble:
        for i in range(4):
            train_loader, valid_loader, test_loader = \
                get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, cfg.data.url,
                                cfg.data.path, i)
            trainer.fit(engine, train_dataloader=train_loader, val_dataloaders=valid_loader)
            trainer.save_checkpoint(f"example_{i}.ckpt")
    else:
        train_loader, valid_loader, test_loader = \
            get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, cfg.data.url,
                            cfg.data.path, cfg.data.seg_reduce)
        trainer.fit(engine, train_dataloader=train_loader, val_dataloaders=valid_loader)
