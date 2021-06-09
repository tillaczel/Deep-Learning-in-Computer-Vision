import os
import numpy as np
import glob
import PIL.Image as Image
from omegaconf import DictConfig
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import zipfile
import gdown
import os
import PIL.Image as Image
import h5py
import pandas as pd

def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in np.array(hdf5_data[name[index][0]])])

def get_metadata(folder='train'):
    mat_data = h5py.File(os.path.join(folder, 'digitStruct.mat'))
    size = mat_data['/digitStruct/name'].size

    data = []
    for _i in tqdm(range(size)):
        pic = get_name(_i, mat_data)
        box = get_box_data(_i, mat_data)
        box["file"] = pic
        data.append(box)
    return pd.DataFrame(data)

def random_square(img, meta):
    h, w, _ = img.shape
    available_modes = []
    leftest = np.min(np.array(meta['left']))
    if leftest > 32:
        available_modes.append(0)
    rightest = np.max(np.array(meta['left']) + np.array(meta['width']))
    if w - rightest > 32:
        available_modes.append(1)
    topest = np.min(np.array(meta['top']))
    if topest > 32:
        available_modes.append(2)
    lowest = np.max(np.array(meta['top']) + np.array(meta['height']))
    if h - lowest > 32:
        available_modes.append(3)

    mode_boundaries = [
        [0, leftest, 0, h],
        [rightest, w, 0, h],
        [0, w, 0, topest],
        [0, w, lowest, h]
    ]
    if len(available_modes) < 0:
        print(meta.file)
        raise ValueError('no possible part')
    mode = np.random.randint(0, len(available_modes))

    l, r, t, b = random_square_from_boundaries(*mode_boundaries[available_modes[mode]])
    return img[t:b, l:r]


def random_square_from_boundaries(l, r, t, b):
    w = r - l
    h = b - t
    size = np.random.randint(32, min(w, h))
    left = l + np.random.randint(0, w - size)
    right = left + size
    top = t + np.random.randint(0, h - size)
    bottom = top + size
    return left, right, top, bottom


class NoDigitDataset(Dataset):
    def __init__(self, transform, folder="./"):
        self.transform = transform
        self.folder = folder
        self.df = get_metadata(folder=folder)
        # TODO: seed

    def __len__(self):
        # 'Returns the total number of samples'
        return len(self.df) # this could be arbitrarily larger

    def __getitem__(self, idx):
        # 'Generates one sample of data'
        meta = self.df.iloc[idx]
        image = Image.open(os.path.join(self.folder, meta.file))
        img = np.array(image)

        X = random_square(img, meta)
        X = self.transform(X)
        return X, 10

def get_data_no_digit():
    # TODO:
    pass

def get_data_svhn(size, train_augmentation, batch_size, base_path: str = './'):
    train_transform, valid_transform = get_transforms(size, train_augmentation)

    # train_set = Hotdog_NotHotdog(train=True, transform=train_transform, base_path=base_path)
    # valid_set = Hotdog_NotHotdog(train=False, transform=valid_transform, base_path=base_path)

    # TODO: get regular SVHN
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader


def get_transforms(size, train_augmentation):
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = list()
    if 'random_crop' in train_augmentation:
        train_transform.append(transforms.Resize((int(1.1*size), int(1.1*size))))
        train_transform.append(transforms.RandomCrop((size, size)))
    else:
        train_transform.append(transforms.Resize((size, size)))
    if 'random_horizontal_flip' in train_augmentation:
        train_transform.append(transforms.RandomHorizontalFlip())
    if 'color_jitter' in train_augmentation:
        train_transform.append(transforms.ColorJitter())
    train_transform.append(transforms.ToTensor())
    train_transform.append(transforms.Normalize(norm_mean, norm_std))
    train_transform = transforms.Compose(train_transform)

    valid_transform = [transforms.Resize((size, size)),
                       transforms.ToTensor(),
                       transforms.Normalize(norm_mean, norm_std)]
    valid_transform = transforms.Compose(valid_transform)
    return train_transform, valid_transform

# def plot_data(loader):
#     images, labels = next(iter(loader))
#     plt.figure(figsize=(20,10))
#
#     for i in range(21):
#         plt.subplot(5,7,i+1)
#         plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
#         plt.title(['hotdog', 'not hotdog'][labels[i].item()])
#         plt.axis('off')
