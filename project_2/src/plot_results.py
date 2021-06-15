import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np


def plot_predictions(dataset, model, device, n=6, current_epoch=None):
    # idxs = np.random.choice(np.arange(len(dataset)), replace=False, size=n)
    idxs = [11, 1329, 769, 835, 134, 983, 97, 404, 439, 1383, 913, 1844, 231, 318, 234, 137, 598][:n]
    _, _, predictions = get_data(dataset, model, device, idxs, mode='single')
    input_data, segmentations, predictions_mc = get_data(dataset, model, device, idxs, mode='mc_dropout')
    # average dropout/ensemble
    predictions_mc = np.mean(predictions_mc, axis=1)
    predictions = np.mean(predictions, axis=1)
    _plot_pred(input_data, segmentations, predictions, predictions_mc,
               n=n, current_epoch=current_epoch)


def _plot_pred(input_data, segmentations, predictions_single, predictions_mc, n=6, current_epoch=None):
    fig, axs = plt.subplots(n, 4, figsize=(18, n*5))
    for i in range(n):
        show_title = i == 0
        plot_subplot(axs[i, 0], input_data[i], 'Input', show_title=show_title)
        plot_subplot(axs[i, 1], segmentations[i], 'Segmentation', show_title=show_title)
        plot_subplot(axs[i, 2], predictions_single[i], 'Prediction', show_title=show_title)
        plot_subplot(axs[i, 3], predictions_mc[i], 'MC Dropout', show_title=show_title)
    plt.subplots_adjust(hspace=0.02)
    plt.subplots_adjust(wspace=0.001)
    fname = 'preds_{current_epoch}.png' if current_epoch is not None else 'preds.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def plot_subplot(ax, img, title, show_title=True):
    ax.imshow(img, cmap="gray")
    if show_title:
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
        images.append(img), segmentations.append(seg), preds.append(pred)

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

