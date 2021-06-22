import os
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pprint

from project_3.src.utils import download_file
from project_3.src.engine import EngineModule
from project_3.src.data import get_dataloaders


def run_eval(cfg: DictConfig):
    # load experiment config
    download_file(cfg.run_id, 'train_config.yaml')
    train_cfg = OmegaConf.load('train_config.yaml')

    cfg_file = os.path.join(wandb.run.dir, 'eval_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file, base_path=wandb.run.dir)  # this will force sync it

    train_loader, valid_loader,  test_dataset_horse, test_dataset_zebra = \
        get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, cfg.data.url,
                        cfg.data.path, samples_per_epoch=cfg.training.samples_per_epoch)

    download_file(cfg.run_id, "model.ckpt")

    engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg, test_dataset_horse=test_dataset_horse, test_dataset_zebra=test_dataset_zebra)
    engine.visualize(n=12)


