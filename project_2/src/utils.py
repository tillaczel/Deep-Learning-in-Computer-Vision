import wandb
import os
import torch

import matplotlib.pyplot as plt
import numpy as np


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

