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
    wandb.save(cfg_file, base_path=wandb.run.dir)  # this will force sync it

    wandb.save('*.ckpt')  # should keep it up to date
    engine = EngineModule(cfg)
    trainer = get_trainer(cfg, engine)

    if cfg.model.ensemble:
        model_path = os.path.join(wandb.run.dir, 'ensemble_models')

        os.mkdir(model_path) if not os.path.isdir(model_path) else None
        for i in range(4):
            engine = EngineModule(cfg)
            trainer = get_trainer(cfg, engine)
            train_loader, valid_loader, test_loader = \
                get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, cfg.data.url,
                                cfg.data.path, i)
            trainer.fit(engine, train_dataloader=train_loader, val_dataloaders=valid_loader)
            model_name = os.path.join(model_path, f"ensemble_model_{i}.ckpt")
            trainer.save_checkpoint(model_name)
            wandb.save(model_name, base_path=model_path)
    else:
        train_loader, valid_loader, test_loader = \
            get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, cfg.data.url,
                            cfg.data.path, cfg.data.seg_reduce)
        trainer.fit(engine, train_dataloader=train_loader, val_dataloaders=valid_loader)
        model_name = os.path.join(wandb.run.dir, f"model.ckpt")
        trainer.save_checkpoint(model_name)
        wandb.save(model_name, base_path=wandb.run.dir)
