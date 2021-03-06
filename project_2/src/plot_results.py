import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np

def plot_uncertainty(dataset, models, n=6, current_epoch=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idxs = [117, 143, 673, 737, 238, 683, 937, 204, 1900, 1383, 913, 1844, 231, 318, 234, 137, 598][:n]
    if len(models) > 1:
        print('Ensemble')
        all_input_data, all_seg, all_preds = get_data_ensemble(dataset, models, device, idxs)
        print('all preds', all_preds.shape, all_preds.min(), all_preds.max())
        _plot_uncertainty(all_input_data, all_seg, all_preds, n=n, suffix="_ensemble", current_epoch=current_epoch)

    print('MC dropout')
    all_input_data, all_seg, all_preds = get_data_mc(dataset, models[0], device, idxs)
    print('all preds', all_preds.shape, all_preds.min(), all_preds.max())
    _plot_uncertainty(all_input_data, all_seg, all_preds, n=n, suffix="_dropout", current_epoch=current_epoch)


def _plot_uncertainty(input_data, all_seg, all_preds, n=6, suffix="", current_epoch=None):
    # Visualise means and variance
    means_preds, means_seg, vars_preds, vars_seg = np.mean(all_preds, axis=1), np.mean(all_seg, axis=1), np.var(
        all_preds, axis=1), np.var(all_seg, axis=1)
    print('before scaling:')
    print('var pred', vars_preds.min(), vars_preds.max())
    print('var seg', vars_seg.min(), vars_seg.max())

    vars_preds = (vars_preds - vars_preds.min()) / (vars_preds.max() - vars_preds.min())
    vars_seg = (vars_seg - vars_seg.min()) / (vars_seg.max() - vars_seg.min())
    print('scaled:')
    print('var pred', vars_preds.min(), vars_preds.max())
    print('var seg', vars_seg.min(), vars_seg.max())
    fig, axs = plt.subplots(n, 5, figsize=(15, n * 5))
    for i in range(n):
        show_title = True
        plot_subplot(axs[i, 0], input_data[i], 'Input', show_title=show_title)
        plot_subplot(axs[i, 1], means_preds[i], 'Mean of predictions', show_title=show_title)
        plot_subplot(axs[i, 2], vars_preds[i], 'Var. of predictions', show_title=show_title, cmap='gist_heat')
        plot_subplot(axs[i, 3], np.squeeze(means_seg[i]), 'Mean of segmentations', show_title=show_title)
        plot_subplot(axs[i, 4], np.squeeze(vars_seg[i]), 'Var. of segmentations', show_title=show_title, cmap='gist_heat')
    fname = f'uncertanty_{current_epoch}{suffix}.png' if current_epoch is not None else f'uncertanty{suffix}.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def plot_predictions(dataset, model, device, n=6, current_epoch=None):
    # idxs = np.random.choice(np.arange(len(dataset)), replace=False, size=n)
    idxs = [117, 143, 673, 737, 238, 683, 937, 204, 1900, 1383, 913, 1844, 231, 318, 234, 137, 598][:n]
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
        # show_title = i == 0
        show_title = True
        plot_subplot(axs[i, 0], input_data[i], 'Original image', show_title=show_title)
        plot_subplot(axs[i, 1], segmentations[i].mean(axis=0), 'Mean Annotation', show_title=show_title)
        plot_subplot(axs[i, 2], predictions_single[i], 'Prediction', show_title=show_title)
        plot_subplot(axs[i, 3], predictions_mc[i], 'MC Dropout', show_title=show_title)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.01)
    fname = f'preds_{current_epoch}.png' if current_epoch is not None else 'preds.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def plot_predictions_ensemble(dataset, models, device, n=6, current_epoch=None):
    # idxs = np.random.choice(np.arange(len(dataset)), replace=False, size=n)
    idxs = [117, 143, 673, 737, 238, 683, 937, 204, 1900, 1383, 913, 1844, 231, 318, 234, 137, 598][:n]
    _, _, predictions = get_data(dataset, models[0], device, idxs, mode='single')
    input_data, segmentations, predictions_mc = get_data(dataset, models[0], device, idxs, mode='mc_dropout')
    _, _, predictions_ens = get_data(dataset, models, device, idxs, mode='ensemble')
    # average dropout/ensemble
    print('seg shape', segmentations.shape)
    predictions_mc = np.mean(predictions_mc, axis=1)
    predictions_ens = np.mean(predictions_ens, axis=1)
    predictions = np.mean(predictions, axis=1)
    _plot_pred_ens(input_data, segmentations, predictions, predictions_mc, predictions_ens,
               n=n, current_epoch=current_epoch)


def _plot_pred_ens(input_data, segmentations, predictions_single, predictions_mc, predictions_ens, n=6, current_epoch=None):
    fig, axs = plt.subplots(n, 5, figsize=(20, n*5))
    for i in range(n):
        # show_title = i == 0
        show_title = True
        plot_subplot(axs[i, 0], input_data[i], 'Original image', show_title=show_title)
        plot_subplot(axs[i, 1], segmentations[i].mean(axis=0), 'Mean Annotation', show_title=show_title)
        plot_subplot(axs[i, 2], predictions_single[i], 'Single model', show_title=show_title)
        plot_subplot(axs[i, 3], predictions_mc[i], 'MC Dropout', show_title=show_title)
        plot_subplot(axs[i, 4], predictions_ens[i], 'Ensemble', show_title=show_title)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.01)
    fname = f'preds_{current_epoch}.png' if current_epoch is not None else 'preds.png'
    fname = os.path.join(wandb.run.dir, fname)
    plt.savefig(fname)
    wandb.save(fname, base_path=wandb.run.dir)


def plot_subplot(ax, img, title, show_title=True, cmap="gray"):
    ax.imshow(img, cmap=cmap)
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
    segmentations = segmentations.detach().cpu().numpy()[:, :, 0] # remove channels dim
    images = images.detach().cpu().numpy()[:, 0] # remove channels dim
    return images, segmentations, preds


def get_data_ensemble(dataset, models, device, idxs):
    images, segmentations, preds = list(), list(), list()
    for idx in idxs:
        img, seg = dataset[idx]
        images.append(img), segmentations.append(seg)
        preds_i = list()
        for i, model in enumerate(models):
            model.to(device)
            model.eval()
            pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))
            preds_i.append(pred)
        preds.append(torch.cat(preds_i, dim=0))
    return refactor_outputs(images, segmentations, preds)

def get_data_single(dataset, model, device, idxs):
    model.to(device)
    model.eval()
    images, segmentations, preds = list(), list(), list()
    for idx in idxs:
        img, seg = dataset[idx]
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))  # Do a forward pass of validation data to get predictions
        images.append(img), segmentations.append(seg), preds.append(pred)

    return refactor_outputs(images, segmentations, preds)


def get_data_mc(dataset, model, device, idxs, n_samples=32):
    model.to(device)
    model.eval_with_dropout()
    images, segmentations, preds = list(), list(), list()
    for idx in idxs:
        img, seg = dataset[idx]
        img_repeated = img.unsqueeze(0).repeat((n_samples, 1, 1, 1))
        pred = torch.sigmoid(model(img_repeated.to(device))).detach().cpu()  # Do a forward pass of validation data to get predictions
        images.append(img), segmentations.append(seg), preds.append(pred)

    return refactor_outputs(images, segmentations, preds)

