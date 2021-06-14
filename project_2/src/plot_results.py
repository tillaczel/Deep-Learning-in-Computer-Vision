import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np


def plot_predictions(dataset, model, device, n=6, current_epoch=None):
    input_data, segmentations, predictions = get_data(dataset, model, device, n)
    print(input_data.shape, segmentations.shape, predictions.shape)
    fig, axs = plt.subplots(3, 5, figsize=(n*5, 15))
    for i in range(6):
        axs[i, 0].imshow(input_data[i], cmap="gray")
        axs[i, 0].set_title(f"Input")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(segmentations[i], cmap="gray")
        axs[i, 1].set_title(f"Segmentation")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(predictions[i], cmap="gray")
        axs[i, 2].set_title(f"Prediction")
        axs[i, 2].axis('off')

    fname = f'preds_{current_epoch}.png' if current_epoch is not None else 'preds.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname)


def get_data(dataset, model, device, n):
    images, segmentations, preds = list(), list(), list()
    for i in range(n):
        img, seg = dataset[i]
        pred = model(img.unsqueeze(0).to(device))[0]  # Do a forward pass of validation data to get predictions
        images.append(img), segmentations.append(seg), preds.append(pred)

    images, segmentations, preds = map(torch.stack, [images, segmentations, preds])
    images, segmentations, preds = [obj.detach().cpu().numpy() for obj in [images, segmentations, preds]]
    images, segmentations, preds = map(np.moveaxis, [images, segmentations, preds], [1, 1, 1], [-1, -1, -1])
    return images, segmentations, preds

