import sys
import os

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from project_2.src.api.eval import run_eval


wandb.init(project='p2', entity='dlcv')


@hydra.main(config_path='config', config_name="default_eval")
def main(cfg : DictConfig):
    run_eval(cfg)


if __name__ == '__main__':
    main()