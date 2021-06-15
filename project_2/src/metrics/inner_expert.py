import torch
from torch.utils.data import DataLoader

from . import calc_all_metrics


def calc_inner_expert(loader):
    loader = DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)
    results = list()
    for img, seg in loader:
        for i in range(3):
            for j in range(i+1, 4):
                results.append(calc_all_metrics(seg[0], seg[1].to(torch.int)))
    results
    print(results)

