import wandb
import os
import torch

import matplotlib.pyplot as plt
import numpy as np


def download_file(run_id, filename):
    api = wandb.Api()
    run = api.run(f"dlcv/p1/{run_id}")
    files = run.files()
    for file in files:
        if file.name == filename:
            file.download(replace=True)
            return
    raise RuntimeError(f"File {filename} not found in dlcv/p1/{run_id}")


def get_state_from_checkpoint(run_id, filename="model.ckpt", replace=True):
  if not os.path.isfile(filename) or replace:
    download_file(run_id, filename)
  chpt = torch.load(filename, map_location=torch.device('cpu'))
  return chpt['state_dict']


def get_heatmap(x, model, normalize=True):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    x.requires_grad = True
    x_noise = x.repeat((128, 1, 1, 1)) + torch.randn([128, 3, 224, 224]) * 0.1
    model.zero_grad()

    if torch.cuda.is_available():
        x_noise = x_noise.cuda()
    pred = model(x_noise)

    pred.sum().backward()
    grad = x.grad.numpy()
    grad = np.abs(grad).sum(axis=0)
    if normalize:
        grad -= grad.min()
        grad /= grad.max()
    model.zero_grad()

    if torch.cuda.is_available():
        x = x.cuda()
    predicted_hotdog = ~(model(x.unsqueeze(0)) > 0).cpu().numpy()[0, 0]

    return grad, predicted_hotdog


def plot_heatmaps(test_dataloader, engine, n_rows=10):
    hotdog, not_hotdog = list(), list()
    for x_batch, y_batch in test_dataloader:
        for idx in range(x_batch.shape[0]):
            x = x_batch[idx]
            y = y_batch[idx]
            if y == 1 and len(not_hotdog) < n_rows:
                not_hotdog.append(x)
            if y == 0 and len(hotdog) < n_rows:
                hotdog.append(x)
            if len(not_hotdog) == n_rows and len(hotdog) == n_rows:
                break
        if len(not_hotdog) == n_rows and len(hotdog) == n_rows:
            break

    fig, axs = plt.subplots(n_rows, 4, figsize=(20, n_rows * 5))
    for i_row, (x_0, x_1) in enumerate(zip(hotdog, not_hotdog)):
        for i_class, x in enumerate([x_0, x_1]):
            grad, predicted_hotdog = get_heatmap(x, engine.model, normalize=True)
            axs[i_row, i_class * 2].imshow(grad, cmap="Greys_r")
            axs[i_row, i_class * 2].set_title(f"Predicted hotdog: {predicted_hotdog}")
            axs[i_row, i_class * 2].axis('off')

            axs[i_row, i_class * 2 + 1].imshow(np.swapaxes(np.swapaxes(x.detach().numpy() / 4 + 0.5, 0, 2), 0, 1))
            axs[i_row, i_class * 2 + 1].set_title("Label: " + ['hotdog', 'not hotdog'][i_class])
            axs[i_row, i_class * 2 + 1].axis('off')

    fname = os.path.join(wandb.run.dir, 'heatmaps.png')
    plt.savefig(fname)
    wandb.save(fname)


def print_class_dist(dataloader, title=None):
    labels = list()
    for _, ys in dataloader:
        labels.extend(ys.numpy().tolist())
    if title is not None:
        print(f'{title}:')
    print({element: labels.count(element) for element in set(labels)})
