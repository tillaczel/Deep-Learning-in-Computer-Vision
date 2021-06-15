import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import calc_all_metrics

def get_mc_preds(loader, model, n_samples=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval_with_dropout()

    loader = DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=2)
    mc_preds = []
    segs = []
    for img, seg in tqdm(loader):
        img_repeated = img.repeat((n_samples, 1, 1, 1))
        pred = torch.sigmoid(model(img_repeated.to(device)))
        pred = pred.detach().cpu()
        mc_preds.append(pred.type(torch.float16))
        segs.append(seg)
    return torch.stack(mc_preds), torch.stack(segs)

def get_regular_preds(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval_with_dropout()

    loader = DataLoader(loader.dataset, batch_size=32, shuffle=False, num_workers=2)
    preds = []
    segs = []
    for img, seg in tqdm(loader):
        pred = torch.sigmoid(model(img.to(device)))
        pred = pred.detach().cpu()
        preds.append(pred.type(torch.float16))
        segs.append(seg)
    return torch.cat(preds, dim=0), torch.cat(segs, dim=0)


