import torch
from torch.utils.data import DataLoader

from . import calc_all_metrics


def calc_inner_expert(loader):
    loader = DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)
    for img, seg in loader:
        result = calc_all_metrics(seg[0], seg[1].to(torch.int))
    print(result)

