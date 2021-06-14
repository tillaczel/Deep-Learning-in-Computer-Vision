import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np


def plot_predictions(dataset, model, device, n=6, current_epoch=None):
    input_data, segmentations, predictions = get_data(dataset, model, device, n)
    fig, axs = plt.subplots(n, 3, figsize=(15, n*5))
    for i in np.random.choice(np.arange(len(dataset), n)):
        plot_subplot(axs[i, 0], input_data[i], 'Input')
        plot_subplot(axs[i, 1], segmentations[i], 'Segmentation')
        plot_subplot(axs[i, 2], predictions[i], 'Prediction')

    fname = f'preds_{current_epoch}.png' if current_epoch is not None else 'preds.png'
    fname = os.path.join(wandb.run.dir, fname)
    print(fname)
    plt.savefig(fname)
    wandb.save(fname)


def plot_subplot(ax, img, title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis('off')


def get_data(dataset, model, device, n):
    images, segmentations, preds = list(), list(), list()
    for i in range(n):
        img, seg = dataset[i]
        pred = model(img.unsqueeze(0).to(device))[0]  # Do a forward pass of validation data to get predictions
        images.append(img), segmentations.append(seg), preds.append(pred)

    images, segmentations, preds = map(torch.stack, [images, segmentations, preds])
    images, segmentations, preds = [obj.detach().cpu().numpy()[:, 0] for obj in [images, segmentations, preds]]
    return images, segmentations, preds

