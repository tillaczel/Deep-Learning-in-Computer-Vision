import sys
import os

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

from project_1_1.src.trainer import get_test_trainer
from project_1_1.src.utils import download_file, plot_heatmaps
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from project_1_1.src.data import get_data
from project_1_1.src.engine import EngineModule

from project_1_1.src.data import download_data

wandb.init(project='p1', entity='dlcv')

@hydra.main(config_path='config', config_name="default_eval")
def eval(cfg : DictConfig):
    # load experiment config
    download_file(cfg.run_id, 'train_config.yaml')
    train_cfg = OmegaConf.load('train_config.yaml')

    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, 'eval_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file) # this will force sync it

    download_data(cfg.data.path)
    _, test_dataloader = get_data(cfg.data.size, cfg.data.train_augmentation, cfg.batch_size,
                                                 base_path=cfg.data.path)

    download_file(cfg.run_id, "model.ckpt")
    engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)

    trainer = get_test_trainer(cfg, engine)
    trainer.validate(engine, val_dataloaders=test_dataloader, ckpt_path=None)
    plot_heatmaps(test_dataloader, engine)

if __name__ == '__main__':
    eval()