import sys
import os

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

from project_1_2.src.trainer import get_test_trainer
from project_1_2.src.utils import download_file, plot_heatmaps
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from project_1_2.src.data import get_data_raw
from project_1_2.src.engine import EngineModule
from torch import nn
import numpy as np


wandb.init(project='p2', entity='dlcv')


def coord_to_boxes(coord, ratio): # TODO: this is probably wrong...
    centers = ((coord * 32 + 112) * ratio).astype(int)
    return centers


@hydra.main(config_path='config', config_name="default_eval")
def eval(cfg : DictConfig):
    # load experiment config
    download_file(cfg.run_id, 'train_config.yaml')
    train_cfg = OmegaConf.load('train_config.yaml')

    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, 'eval_config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file) # this will force sync it

    resize_ratios = (0.5, 0.7, 0.9) # todo: config
    # dataset = get_data_raw(split='test', resize_ratios=resize_ratios)
    dataset = get_data_raw(split='train', resize_ratios=resize_ratios)

    download_file(cfg.run_id, "model.ckpt")
    engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)

    # this will go to separate func
    engine.model.resnet[-1] = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
    n_images_to_process = 1
    threshold = 0.2

    for i in range(n_images_to_process):
        all_frames = []
        resized, original_image, ratios, meta = dataset[0]
        for img, ratio in zip(resized, ratios):
            y = engine(img.unsqueeze(dim=0))
            input_max, input_indexes = torch.max(torch.softmax(y, 1), 1)
            mask = (input_max > threshold) & (input_indexes != 10)
            coordinates = np.argwhere(mask.cpu().numpy())[:, 1]
            probs = input_max[mask].cpu().numpy()
            pred_digits  = input_indexes[mask].cpu().numpy()

            # TODO: coordinates to frames (this is probably wrong)
            size = int(ratio * 224)
            centers = coord_to_boxes(coordinates, ratio)
            # TODO: this is shit
            for center in centers:
                all_frames.append((center, size))


            # this will have the output coordinates, probabilities and the predicted labels
            # as a list of tuples
            frames = list(zip(coordinates, probs, pred_digits))
            print(frames)

            # TODO: reject boxes
            # TODO: print image



if __name__ == '__main__':
    eval()