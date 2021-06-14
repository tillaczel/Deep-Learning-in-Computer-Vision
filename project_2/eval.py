import sys
import os


sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

from project_1_2.src.trainer import get_test_trainer
from project_1_2.src.bbox import plt_bboxes, filter_bboxes, run_detection
from project_1_2.src.utils import download_file, plot_heatmaps
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from tqdm import tqdm
from project_1_2.src.data import get_data_raw
from project_1_2.src.engine import EngineModule
from torch import nn
import numpy as np


wandb.init(project='p2', entity='dlcv')




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


    download_file(cfg.run_id, "model.ckpt")
    engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)
    run_detection(engine)
    # this will go to separate func


if __name__ == '__main__':
    eval()