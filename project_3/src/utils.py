import wandb
import os
import torch
from tqdm import tqdm

from project_2.src.engine import EngineModule


def download_file(run_id, filename):
    api = wandb.Api()
    run = api.run(f"dlcv/p3/{run_id}")
    files = run.files()
    for file in files:
        if file.name == filename:
            file.download(replace=True)
            return
    raise RuntimeError(f"File {filename} not found in dlcv/p3/{run_id}")


def get_state_from_checkpoint(run_id, filename="model.ckpt", replace=True):
    if not os.path.isfile(filename) or replace:
        download_file(run_id, filename)
    chpt = torch.load(filename, map_location=torch.device('cpu'))
    return chpt['state_dict']


def get_ensemble_models(run_id, train_config, n_checkpoints=4):
    models = list()
    for i in tqdm(range(n_checkpoints), desc='Downloading models'):
        download_file(run_id, f"ensemble_model_{i}.ckpt")
        engine = EngineModule.load_from_checkpoint(f"ensemble_model_{i}.ckpt", config=train_config)
        models.append(engine.model)

    return models
