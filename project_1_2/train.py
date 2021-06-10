import sys
import os

from project_1_2.src.bbox import run_detection

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])
from pytorch_lightning.trainer.supporters import CombinedLoader

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from project_1_2.src.data import get_data_no_digit, get_data_svhn, get_dataloaders
from project_1_2.src.engine import EngineModule
from project_1_2.src.trainer import get_trainer
from project_1_2.src.utils import print_class_dist

wandb.init(project='p2', entity='dlcv')

@hydra.main(config_path='config', config_name="default")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_file = os.path.join(wandb.run.dir, 'train_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)  # this will force sync it

    train_loader, valid_loader = get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, overlap=cfg.overlap)

    # print_class_dist(train_loader, title='Train set'), print_class_dist(valid_loader, title='Valid no_digit set')

    engine = EngineModule(cfg)

    wandb.save('*.ckpt')  # should keep it up to date

    trainer = get_trainer(cfg, engine)

    trainer.fit(engine, train_dataloader=train_loader, val_dataloaders=valid_loader)

    run_detection(engine)

    # TODO: visualizations


if __name__ == '__main__':
    run_training()


