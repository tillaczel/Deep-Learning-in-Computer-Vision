import torch
from torch.utils.data import DataLoader

from . import calc_all_metrics


def calc_inner_expert(loader):
    loader = DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)
    results = dict()
    for img, seg in loader:
        for i in range(3):
            for j in range(i+1, 4):
                result = calc_all_metrics(seg[0], seg[1].to(torch.int))
                for v, k in result.items():
                    print(v)
                    if k in results.keys():
                        results[k] += v/6
                    else:
                        results[k] = v/6
    print(results)

