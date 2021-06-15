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
    results = get_metrics(seg, seg)
    print('Inner expert', results)
    return results


def calc_mean(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    preds, segs = list(), list()
    for img, seg in tqdm(loader):
        pred = torch.sigmoid(model(img.to(device)))
        pred = pred.detach().cpu()
        preds.append(pred.type(torch.float16))
        segs.append(seg)
    preds, segs = torch.cat(preds, dim=0), torch.cat(segs, dim=0)
    print(preds.shape, segs.shape)
    results = get_metrics(preds, segs)
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
