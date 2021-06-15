import torch
from torch.utils.data import DataLoader

from . import calc_all_metrics

def get_mc_preds(loader, model, n_samples=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval_with_dropout()

    loader = DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=2)
    mc_preds = []
    segs = []
    for img, seg in loader:
        img_repeated = img.repeat((n_samples, 1, 1, 1))
        pred = torch.sigmoid(model(img_repeated.to(device)))
        pred = pred.detach().cpu()
        mc_preds.append(pred)
        segs.append(seg)
        print(len(mc_preds))
    return torch.stack(mc_preds), torch.stack(segs)





