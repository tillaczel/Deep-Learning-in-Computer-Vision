import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np


def plot_predictions(dataset, model, device, n=6, current_epoch=None):
    idxs = np.random.choice(np.arange(len(dataset)), replace=False, size=n)
    input_data, segmentations, predictions = get_data(dataset, model, device, idxs)
    _plot_pred(input_data, segmentations, predictions, n=n, current_epoch=current_epoch)


def _plot_pred(input_data, segmentations, predictions, n=6, current_epoch=None):
    fig, axs = plt.subplots(n, 3, figsize=(15, n*5))
    for i in range(n):
        plot_subplot(axs[i, 0], input_data[i], 'Input')
        plot_subplot(axs[i, 1], segmentations[i], 'Segmentation')
        plot_subplot(axs[i, 2], predictions[i], 'Prediction')

    fname = f'preds_{current_epoch}.png' if current_epoch is not None else 'preds.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def plot_subplot(ax, img, title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis('off')


def get_data(dataset, model, device, idxs):
    images, segmentations, preds = list(), list(), list()
    for idx in idxs:
        img, seg = dataset[idx]
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device))[0])  # Do a forward pass of validation data to get predictions
        images.append(img), segmentations.append(seg), preds.append(pred)

    images, segmentations, preds = map(torch.stack, [images, segmentations, preds])
    images, segmentations, preds = [obj.detach().cpu().numpy()[:, 0] for obj in [images, segmentations, preds]]
    return images, segmentations, preds

