import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import calc_all_metrics


def get_data(loader):
    loader = DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)
    img, seg = next(iter(loader))
    return img, seg


def calc_inner_expert(loader):
    img, seg = get_data(loader)
    results = get_metrics(seg, seg, one_pred=False)
    return results


def get_metrics(pred, seg, one_pred=True):
    results = dict()
    if one_pred:
        for j in range(1, 4):
            result = calc_all_metrics(pred[:, 0], seg[:, j].to(torch.int))
            for k, v in result.items():
                if k in results.keys():
                    results[k] += v / 3
                else:
                    results[k] = v / 3
    else:
        for i in range(pred.shape[1]):
            for j in range(1 + i, 4):
                result = calc_all_metrics(pred[:, i], seg[:, j].to(torch.int))
                for k, v in result.items():
                    if k in results.keys():
                        results[k] += v / 6
                    else:
                        results[k] = v / 6
    return results
