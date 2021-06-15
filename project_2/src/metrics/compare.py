import torch
from torch.utils.data import DataLoader

from . import calc_all_metrics


def calc_inner_expert(loader):
    loader = DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)
    img, seg = next(iter(loader))
    import matplotlib.pyplot as plt
    for i in range(10):
        a = torch.mean(seg[i], 0, keepdim=True)
        plt.imshow(a[0, 0])
        plt.show()
    results = get_metrics(seg, seg)
    print('Inner expert', results)
    return results


def get_metrics(pred, seg):
    results = dict()
    for i in range(pred.shape[1]):
        for j in range(i+1, 4):
            result = calc_all_metrics(pred[:, i], seg[:, j].to(torch.int))
            for k, v in result.items():
                if k in results.keys():
                    results[k] += v/6
                else:
                    results[k] = v/6
    return results
