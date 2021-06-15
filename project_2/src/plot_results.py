import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np


def plot_predictions(dataset, model, device, n=6, current_epoch=None, mode='single'):
    idxs = np.random.choice(np.arange(len(dataset)), replace=False, size=n)
    input_data, segmentations, predictions = get_data(dataset, model, device, idxs, mode=mode)
    # average dropout/ensemble (single will have
    predictions = np.mean(predictions, axis=1)
    _plot_pred(input_data, segmentations, predictions, mode=mode, n=n, current_epoch=current_epoch)


def _plot_pred(input_data, segmentations, predictions, mode, n=6, current_epoch=None):
    fig, axs = plt.subplots(n, 3, figsize=(15, n*5))
    for i in range(n):
        plot_subplot(axs[i, 0], input_data[i], 'Input')
        plot_subplot(axs[i, 1], segmentations[i], 'Segmentation')
        plot_subplot(axs[i, 2], predictions[i], 'Prediction')

    fname = f'preds_{current_epoch}_{mode}.png' if current_epoch is not None else f'preds_{mode}.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def plot_subplot(ax, img, title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis('off')


def get_data(dataset, model, device, idxs, mode, mode_config=None):
    if mode_config is None: # use default kwargs
        mode_config = {}

    if mode == 'single':
        return get_data_single(dataset, model, device, idxs, **mode_config)
    elif mode == 'mc_dropout':
        return get_data_mc(dataset, model, device, idxs, **mode_config)
    elif mode == 'ensemble':
        return get_data_ensemble(dataset, model, device, idxs, **mode_config)
    else:
        raise ValueError(f'Mode must be in [single, mc_dropout, ensemble]. Found: {mode}')

def refactor_outputs(images, segmentations, preds):
    images, segmentations, preds = map(torch.stack, [images, segmentations, preds])
    preds = preds.detach().cpu().numpy()[:, :, 0] # remove channels dim
    images, segmentations = [obj.detach().cpu().numpy()[:, 0] for obj in [images, segmentations]]
    return images, segmentations, preds


def get_data_ensemble(dataset, model, device, idxs):
    raise NotImplementedError


def get_data_single(dataset, model, device, idxs):
    images, segmentations, preds = list(), list(), list()
    for idx in idxs:
        img, seg = dataset[idx]
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))  # Do a forward pass of validation data to get predictions
        images.append(img.unsqueeze(0)), segmentations.append(seg), preds.append(pred)

    return refactor_outputs(images, segmentations, preds)


def get_data_mc(dataset, model, device, idxs, n_samples=64):
    model.eval_with_dropout()
    images, segmentations, preds = list(), list(), list()
    for idx in idxs:
        img, seg = dataset[idx]
        img_repeated = img.unsqueeze(0).repeat((n_samples, 1, 1, 1))
        pred = torch.sigmoid(model(img_repeated.to(device)))  # Do a forward pass of validation data to get predictions
        images.append(img), segmentations.append(seg), preds.append(pred)

    return refactor_outputs(images, segmentations, preds)

