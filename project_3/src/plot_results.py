import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np

def make_plots(dataset, modelA2B, modelB2A, device, n=4, current_epoch=None, suffix=''): # n = number of plots to show
    images, predsA2B, predsB2A = get_data(dataset, modelA2B, modelB2A, device, n)
    plot_images_and_predictions(images, predsA2B, predsB2A, current_epoch=current_epoch, n=n, suffix=suffix)

def plot_subplot(ax, img, title, show_title=True):
    ax.imshow(img)
    if show_title:
        ax.set_title(title)
    ax.axis('off')

def plot_images_and_predictions(input_images, predsA2B, predsB2A, current_epoch=None, n=4, suffix=''):
    fig, axs = plt.subplots(n, 3, figsize=(15, n * 5))
    for i in range(n):
        show_title = True
        plot_subplot(axs[i, 0], input_images[i].moveaxis(0,-1) / 2 + 0.5, 'Input image', show_title=show_title)
        plot_subplot(axs[i, 1], predsA2B[i][0].moveaxis(0,-1) / 2 + 0.5, 'Predicted (A2B) image', show_title=show_title)
        plot_subplot(axs[i, 2], predsB2A[i][0].moveaxis(0,-1) / 2 + 0.5, 'Reconstructed (B2A) image', show_title=show_title)

    fname = f'converted_images{suffix}_{current_epoch}.png' if current_epoch is not None else f'converted_images{suffix}.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def get_data(dataset, modelA2B, modelB2A, device, n=4):
    modelA2B.to(device), modelB2A.to(device)
    modelA2B.eval(), modelB2A.eval()
    images, predsA2B, predsB2A  = list(), list(), list()
    idxs = np.random.choice(np.arange(120), replace=False, size=n)
    for idx in idxs:
        img = dataset[idx]
        print('\n img:', img.min(), img.max(), torch.std(img), torch.mean(img))
        predA2B = modelA2B(img.unsqueeze(0).to(device)) # Do a forward pass A2B
        predB2A = modelB2A(predA2B.to(device)) # Forward pass B2A (reconstruct image)
        print('a2b', predA2B.min(), predA2B.max(), torch.std(predA2B), torch.mean(predA2B))
        print('b2a', predB2A.min(), predB2A.max(), torch.std(predB2A), torch.mean(predB2A))
        images.append(img), predsA2B.append(predA2B.detach().cpu()), predsB2A.append(predB2A.detach().cpu())

    return images, predsA2B, predsB2A



