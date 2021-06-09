import sys
import os

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from project_1_2.src.data import get_data_no_digit, get_data_svhn
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

    # download_data(cfg.data.path)
    train_dataloader_svhn, test_dataloader_svhn = get_data_svhn(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size,
                                                 base_path=cfg.data.path)
    train_dataloader_no_dig, test_dataloader_no_dig =  get_data_no_digit(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size,
                                                 base_path=cfg.data.path)

    train_dataloaders = {
        'svhn': train_dataloader_svhn,
        'no_digit': train_dataloader_no_dig
    }

    val_dataloaders = {
        'svhn': test_dataloader_svhn,
        'no_digit': test_dataloader_no_dig
    }

    print_class_dist(train_dataloader, title='Train set'), print_class_dist(test_dataloader, title='Valid set')

    engine = EngineModule(cfg)

    wandb.save('*.ckpt')  # should keep it up to date

    trainer = get_trainer(cfg, engine)

    trainer.fit(engine, train_dataloader=train_dataloaders, val_dataloaders=val_dataloaders)

    # TODO: visualizations


if __name__ == '__main__':
    run_training()


