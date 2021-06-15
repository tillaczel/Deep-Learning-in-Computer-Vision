import os

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from project_2.src.metrics import calc_all_metrics
from project_2.src.metrics.eval_mc import get_mc_preds, get_regular_preds
from project_2.src.utils import download_file, get_ensemble_models
from project_2.src.engine import EngineModule
from project_2.src.data import get_dataloaders
from project_2.src.metrics.compare import calc_inner_expert, calc_mean


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

    calc_inner_expert(test_loader)

    download_file(cfg.run_id, "model.ckpt")
    engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)

    #calc_mean(test_loader, engine.model)
    #calc_inner_expert(test_loader)

    if cfg.is_ensemble:
        print(get_ensemble_models(cfg.run_id, train_cfg))
    else:
        download_file(cfg.run_id, "model.ckpt")
        engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)
        mc_preds, segs = get_mc_preds(test_loader, engine.model, n_samples=32)
        print("\n preds, segs", mc_preds.shape, segs.shape)

        # tODO: make this nicer

        single_preds, segs2 = get_regular_preds(test_loader, engine.model)
        print("\n single preds, segs", single_preds.shape, segs2.shape)

        print(calc_all_metrics(torch.mean(mc_preds, dim=1), torch.mean(segs, dim=1)))
        print(calc_all_metrics(torch.mean(single_preds, dim=1), torch.mean(segs2, dim=1)))



