import sys
import os

sys.path.append('git_repo')

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from project_1_1.src.data import get_data, download_data
from project_1_1.src.engine import EngineModule
from project_1_1.src.trainer import get_trainer

wandb.init(project='p1', entity='dlcv')


@hydra.main(config_path='config', config_name="default")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_file = os.path.join(wandb.run.dir, 'config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)  # this will force sync it

    download_data(cfg.data.path)
    train_dataloader, test_dataloader = get_data(cfg.data.size, cfg.training.batch_size, base_path=cfg.data.path)
    engine = EngineModule(cfg)

    wandb.save('*.ckpt')  # should keep it up to date

    trainer = get_trainer(cfg, engine)

    trainer.fit(engine, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)

    # TODO: visualizations


if __name__ == '__main__':
    run_training()


