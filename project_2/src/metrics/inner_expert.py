import torch

from . import calc_all_metrics


def calc_inner_expert(loader):
    loader.batch_size = len(loader.dataset)
    for img, seg in loader:
        result = calc_all_metrics(seg[0], seg[1].to(torch.int))
    print(result)

