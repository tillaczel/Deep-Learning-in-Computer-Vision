import os
import wandb
from omegaconf import DictConfig, OmegaConf

from project_2.src.metrics.eval_mc import get_mc_preds
from project_2.src.utils import download_file
from project_2.src.engine import EngineModule
from project_2.src.data import get_dataloaders
from project_2.src.metrics.inner_expert import calc_inner_expert


def run_eval(cfg: DictConfig):
    # load experiment config
    download_file(cfg.run_id, 'train_config.yaml')
    train_cfg = OmegaConf.load('train_config.yaml')

    cfg_file = os.path.join(wandb.run.dir, 'eval_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file, base_path=wandb.run.dir)  # this will force sync it

    train_loader, valid_loader, test_loader = \
        get_dataloaders(train_cfg.data.size, train_cfg.data.train_augmentation, train_cfg.training.batch_size,
                        train_cfg.data.url, train_cfg.data.path, seg_reduce='all')

    # calc_inner_expert(test_loader)

    # if cfg.is_ensemble:
    #     raise NotImplementedError
    # else:
    #     download_file(cfg.run_id, "model.ckpt")
    #     engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)
    #     mc_preds, segs = get_mc_preds(test_loader, engine.model, n_samples=32)





