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


def plot_row(x, grad, y, predicted_hotdog, row, n_rows):
    plt.subplot(n_rows, 2, row * 2 + 1)
    plt.imshow(grad, cmap="Greys_r")
    plt.title(f"Predicted hotdog: {predicted_hotdog}")
    plt.axis('off')

    plt.subplot(n_rows, 2, row * 2 + 2)
    plt.imshow(np.swapaxes(np.swapaxes(x[0].detach().numpy()/4 + 0.5, 0, 2), 0, 1))
    plt.title("Label: " + ['hotdog', 'not hotdog'][y.item()])
    plt.axis('off')


def get_heatmap(x, model, normalize=True):
    model.train()
    if torch.cuda.is_available():
      model.cuda()
    x.requires_grad = True
    x_noise = x.repeat((128, 1, 1, 1)) + torch.randn([128, 3, 224, 224]) * 0.1
    model.zero_grad()

    if torch.cuda.is_available():
      x_noise = x_noise.cuda()
    pred = model(x_noise)

    pred.sum().backward()
    predicted_hotdog = (pred.sum() > 0).cpu().numpy()
    grad = x.grad[0].numpy()
    grad = np.abs(grad).sum(axis=0)
    if normalize:
      grad -= grad.min()
      grad /= grad.max()
    model.zero_grad()

    return grad, predicted_hotdog


def plot_heatmaps(test_dataloader, engine):
    plt.figure(figsize=(10, n_rows * 5))
    for x_batch, y_batch in test_dataloader:
        for idx in x_batch.shape[0]:
            x = x_batch[idx]
            y = y_batch[idx]
            print(y)
            # grad, predicted_hotdog = get_heatmap(x, engine.model, normalize=True)
            # plot_row(x, grad, y, predicted_hotdog, i, n_rows)
    # fname = os.path.join(wandb.run.dir, 'heatmaps.png')
    # plt.savefig(fname)
    # wandb.save(fname)
