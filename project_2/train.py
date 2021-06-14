import sys
import os

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])


import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from project_2.src.data import get_dataloaders
from project_2.src.engine import EngineModule
from project_2.src.trainer import get_trainer

wandb.init(project='p3', entity='dlcv')

@hydra.main(config_path='config', config_name="default")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_file = os.path.join(wandb.run.dir, 'train_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)  # this will force sync it

    train_loader, valid_loader, test_loader = \
        get_dataloaders(cfg.data.size, cfg.data.train_augmentation, cfg.training.batch_size, cfg.data.url, cfg.data.path)

    # print_class_dist(train_loader, title='Train set'), print_class_dist(valid_loader, title='Valid no_digit set')

    engine = EngineModule(cfg)

    wandb.save('*.ckpt')  # should keep it up to date

    trainer = get_trainer(cfg, engine)

    trainer.fit(engine, train_dataloader=train_loader, val_dataloaders=valid_loader)



if __name__ == '__main__':
    run_training()


