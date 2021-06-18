import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np

def make_plots(dataset, modelA2B, modelB2A, device, n=4): # n = number of plots to show
    images, predsA2B, predsB2A = get_data(dataset, modelA2B, modelB2A, device, n)
    for img, predA2B, predB2A in zip(images, predsA2B, predsB2A):
        plot_images_and_predictions(img, predA2B, predB2A, current_epoch=None)

def plot_subplot(ax, img, title, show_title=True):
    ax.imshow(img)
    if show_title:
        ax.set_title(title)
    ax.axis('off')

def plot_images_and_predictions(input_images, predsA2B, predsB2B, current_epoch=None, n=4):
    fig, axs = plt.subplots(n, 3, figsize=(15, n * 5))
    for i in range(n):
        show_title = True
        plot_subplot(axs[i, 0], input_images[i], 'Input image', show_title=show_title)
        plot_subplot(axs[i, 1], predsA2B[i], 'Predicted (A2B) image', show_title=show_title)
        plot_subplot(axs[i, 2], predsB2B[i], 'Reconstructed (B2A) image', show_title=show_title)

    fname = f'convertedImages_{current_epoch}.png' if current_epoch is not None else 'convertedImages.png'
    #fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    #wandb.save(fname, base_path=wandb.run.dir)


def get_data(dataset, modelA2B, modelB2A, device, n=4):
    modelA2B.to(device), modelB2A.to(device)
    modelA2B.eval(), modelB2A.eval()
    images, predsA2B, predsB2A  = list(), list(), list()
    idxs = np.random.choice(np.arange(len(dataset)), replace=False, size=n)
    for idx in idxs:
        img, labels = dataset[idx]
        predA2B = modelA2B(img.unsqueeze(0).to(device)) # Do a forward pass A2B
        predB2A = modelB2A(predA2B.unsqueeze(0).to(device)) # Forward pass B2A (reconstruct image)
        images.append(img), predsA2B.append(predA2B), predsB2A.append(predB2A)

    return images, predsA2B, predsB2A
    #return refactor_outputs(images, preds)

#def refactor_outputs(images, segmentations, preds):
#    images, preds = map(torch.stack, [images, preds])
#    preds = preds.detach().cpu().numpy()[:, :, 0] # remove channels dim
#    images = images.detach().cpu().numpy()[:, 0] # remove channels dim
#    return images, segmentations, preds



