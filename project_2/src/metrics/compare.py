import torch
from torch.utils.data import DataLoader

from . import calc_all_metrics


def calc_inner_expert(loader):
    loader = DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)
    img, seg = next(iter(loader))
    results = get_metrics(seg, seg)
    print('Inner expert', results)
    return results


def get_metrics(pred, seg):
    results = dict()
    for i in range(pred.shape[1]):
        for j in range(i+1, 4):
            result = calc_all_metrics(pred[:, i], seg[:, j].to(torch.int))
            print(result)
            for k, v in result.items():
                if k in results.keys():
                    results[k] += v/6
                else:
                    results[k] = v/6
    return results
