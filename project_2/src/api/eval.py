import os
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pprint

from project_2.src.metrics import calc_all_metrics
from project_2.src.metrics.energy import calculate_energy
from project_2.src.metrics.get_preds import get_mc_preds, get_regular_preds, get_ensemble_preds
from project_2.src.utils import download_file, get_ensemble_models
from project_2.src.engine import EngineModule
from project_2.src.data import get_dataloaders
from project_2.src.metrics.compare import calc_inner_expert, get_metrics


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

    print("Inner expert scores:")
    pprint.pprint(calc_inner_expert(test_loader))

    if train_cfg.model.ensemble:
        models = get_ensemble_models(cfg.run_id, train_cfg)
        preds, segs = get_ensemble_preds(test_loader, models)
        print("Ensemble scores:")
        pprint.pprint(get_metrics(preds, segs, one_pred=False))
        del preds, segs, models

    else:
        download_file(cfg.run_id, "model-v1.ckpt")
        engine = EngineModule.load_from_checkpoint("model-v1.ckpt", config=train_cfg)
        preds, segs = get_mc_preds(test_loader, engine.model, n_samples=16)
        print("MC scores:")
        pprint.pprint(get_metrics(preds, segs))
        calculate_energy(preds[:, :, 0], segs)
        del preds, segs

        preds, segs = get_regular_preds(test_loader, engine.model)
        print("Regular scores:")
        print(preds.shape, segs.shape)
        pprint.pprint(get_metrics(preds, segs))
        del preds, segs
