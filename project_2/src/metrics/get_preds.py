import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import calc_all_metrics


def get_mc_preds(loader, model, n_samples=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval_with_dropout()

    loader = DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=2)
    mc_preds, segs = list(), list()
    for img, seg in tqdm(loader, desc='Getting preds'):
        img_repeated = img.repeat((n_samples, 1, 1, 1))
        pred = torch.sigmoid(model(img_repeated.to(device)))
        pred = pred.detach().cpu()
        mc_preds.append(pred.type(torch.float16))
        segs.append(seg)
    model.eval()
    return torch.stack(mc_preds)[:, :, 0], torch.cat(segs, dim=0)


def get_regular_preds(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    preds, segs = list(), list()
    for img, seg in tqdm(loader, desc='Getting preds'):
        pred = torch.sigmoid(model(img.to(device)))
        pred = pred.detach().cpu()
        preds.append(pred.type(torch.float16))
        segs.append(seg)
    return torch.cat(preds, dim=0), torch.cat(segs, dim=0)


def get_ensemble_preds(loader, models):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    segs = next(iter(DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=False, num_workers=2)))[1]
    preds = list()
    for i, model in enumerate(models):
        model.to(device)
        model.eval()
        preds_i = list()
        for img, seg in tqdm(loader, desc=f'Getting preds {i}'):
            pred = torch.sigmoid(model(img.to(device)))
            pred = pred.detach().cpu()
            preds_i.append(pred.type(torch.float16))
        preds.append(torch.cat(preds_i, dim=0))
    preds = torch.stack(preds, dim=1)
    print(preds.shape, segs.shape)
    return None
