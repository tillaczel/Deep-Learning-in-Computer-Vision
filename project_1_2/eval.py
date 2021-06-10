import sys
import os


sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

from project_1_2.src.trainer import get_test_trainer
from project_1_2.src.bbox import plt_bboxes, filter_bboxes
from project_1_2.src.utils import download_file, plot_heatmaps
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from tqdm import tqdm
from project_1_2.src.data import get_data_raw
from project_1_2.src.engine import EngineModule
from torch import nn
import numpy as np


wandb.init(project='p2', entity='dlcv')


def coord_to_boxes(coord, ratio):  # TODO: this is probably wrong...
    center = (coord * 32 + 112)
    coord = ( # who knows where is up or left
        int((center[0] - 112) / ratio),
        int((center[0] + 112) / ratio),
        int((center[1] - 112) / ratio),
        int((center[1] + 112) / ratio),
    )
    return coord


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

    resize_ratios = (0.55, 0.7, 0.8) # todo: config
    # dataset = get_data_raw(split='test', resize_ratios=resize_ratios)
    dataset = get_data_raw(split='train', resize_ratios=resize_ratios)

    download_file(cfg.run_id, "model.ckpt")
    engine = EngineModule.load_from_checkpoint("model.ckpt", config=train_cfg)

    # this will go to separate func
    engine.model.resnet[-1] = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
    n_images_to_process = 10

    thresholds = [(0.1, 0.5), (0.2, 0.5), (0.1, 0.3), (0.25, 0.25)]

    for i in tqdm(range(n_images_to_process)):
        im_dir = os.path.join(wandb.run.dir, f'img_{i}')
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        result2 = []
        resized, original_image, ratios, meta = dataset[i]
        # this is ugly but should work
        for img, ratio in zip(resized, ratios):
            result = []

            y = engine(img.unsqueeze(dim=0))
            probs = torch.softmax(y, 1).detach().cpu().numpy()
            for i in range(probs.shape[-1]):  # x-axis
                for j in range(probs.shape[-2]):  # y-axis
                    p = probs[0, :, j, i]
                    coord = coord_to_boxes(np.array([i, j]), ratio)

                    result.append((
                        coord,
                        p[:10]
                    ))
                    result2.append((
                        coord,
                        p[:10]
                    ))
            for th in thresholds:
                th_dir = os.path.join(im_dir, f'thresh_{th[0]:.2f}_{th[1]:.2f}')
                if not os.path.isdir(th_dir):
                    os.mkdir(th_dir)
                p_threshold, iou_threshold = th
                filename = os.path.join(th_dir, f'bbox_{ratio:.2f}.png')
                filter_bboxes(result, original_image, filename=filename, p_threshold=p_threshold, iou_threshold=iou_threshold)
        for  th in thresholds:
            th_dir = os.path.join(im_dir, f'thresh_{th[0]:.2f}_{th[1]:.2f}')
            if not os.path.isdir(th_dir):
                os.mkdir(th_dir)
            p_threshold, iou_threshold = th
            filename = os.path.join(th_dir, f'bbox_.png')
            filter_bboxes(result2, original_image, filename=filename, p_threshold=p_threshold, iou_threshold=iou_threshold)



if __name__ == '__main__':
    eval()