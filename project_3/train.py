import sys
import os
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

from project_3.src.api.train import run_training

wandb.init(project='p3', entity='dlcv')


@hydra.main(config_path='config', config_name="default_train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == '__main__':
    main()
