import matplotlib.pyplot as plt
import torch
import os
import wandb


def plot_predictions(dataset, model, device, n=6):
    input_data, segmentations, predictions = get_data(dataset, model, device, n=n)

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

    fname = os.path.join(wandb.run.dir, 'preds.png')
    plt.savefig(fname)
    wandb.save(fname)


def get_data(dataset, model, device, n):
    images, segmentations, preds = list(), list(), list()
    for i in range(n):
        img, seg = dataset[i]
        pred = model(img.to(device))  # Do a forward pass of validation data to get predictions
        images.append(img), segmentations.append(seg), preds.append(pred)
    images, segmentations, preds = torch.stack(images), torch.stack(segmentations), torch.stack(preds)

    return images.detach().cpu().numpy(), segmentations.detach().cpu().numpy(), preds.detach().cpu().numpy()

