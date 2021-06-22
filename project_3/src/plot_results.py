import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np

def make_plots(dataset, modelA2B, modelB2A, device, n=4, current_epoch=None, suffix=''): # n = number of plots to show
    images, predsA2B, predsB2A, identity_preds = get_data(dataset, modelA2B, modelB2A, device, n)
    plot_images_and_predictions(images, predsA2B, predsB2A, identity_preds, current_epoch=current_epoch, n=n, suffix=suffix)

def plot_subplot(ax, img, title, show_title=True):
    ax.imshow(img)
    if show_title:
        ax.set_title(title)
    ax.axis('off')

def plot_images_and_predictions(input_images, predsA2B, predsB2A, identity_preds, current_epoch=None, n=4, suffix=''):
    fig, axs = plt.subplots(n, 4, figsize=(16, n * 5))
    for i in range(n):
        show_title = True
        plot_subplot(axs[i, 0], input_images[i].moveaxis(0,-1) / 2 + 0.5, 'Input image', show_title=show_title)
        plot_subplot(axs[i, 1], predsA2B[i][0].moveaxis(0,-1) / 2 + 0.5, 'Predicted (A2B) image', show_title=show_title)
        plot_subplot(axs[i, 2], predsB2A[i][0].moveaxis(0,-1) / 2 + 0.5, 'Reconstructed (B2A) image', show_title=show_title)
        plot_subplot(axs[i, 3], identity_preds[i][0].moveaxis(0,-1) / 2 + 0.5, 'Identity', show_title=show_title)

    fname = f'converted_images{suffix}_{current_epoch}.png' if current_epoch is not None else f'converted_images{suffix}.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def get_data(dataset, modelA2B, modelB2A, device, n=4):
    modelA2B.to(device), modelB2A.to(device)
    modelA2B.eval(), modelB2A.eval()
    images, predsA2B, predsB2A, identity_preds  = list(), list(), list(), list()
    # idxs = np.random.choice(np.arange(120), replace=False, size=n)
    idxs = [1, 3, 6, 12, 58, 2, 34, 4, 68, 100, 99, 89, 78, 11, 23, 21, 16, 90, 83, 45, 43, 47][:n]
    for idx in idxs:
        img = dataset[idx]
        predA2B = modelA2B(img.unsqueeze(0).to(device)) # Do a forward pass A2B
        predB2A = modelB2A(predA2B.to(device)) # Forward pass B2A (reconstruct image)
        identity_pred = modelB2A(img.unsqueeze(0).to(device)) # Forward pass B2A (reconstruct image)
        images.append(img), predsA2B.append(predA2B.detach().cpu()), \
            predsB2A.append(predB2A.detach().cpu()), identity_preds.append(identity_pred.detach().cpu())

    return images, predsA2B, predsB2A, identity_preds



