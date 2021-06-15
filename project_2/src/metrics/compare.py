import torch
from torch.utils.data import DataLoader

from . import calc_all_metrics


def calc_inner_expert(loader):
    loader = DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)
    for img, seg in loader:
        results = get_metrics(seg, seg)
    import numpy as np
    seg = torch.tensor(np.ones((1, 1, 100, 100)))
    seg[:, :, 50] = 0
    results = get_metrics(seg, seg.T)
    print('Inner expert', results)
    return results


def get_metrics(pred, seg):
    results = dict()
    for i in range(3):
        for j in range(i+1, 4):
            result = calc_all_metrics(pred[0], seg[1].to(torch.int))
            for k, v in result.items():
                if k in results.keys():
                    results[k] += v/6
                else:
                    results[k] = v/6
    return results
