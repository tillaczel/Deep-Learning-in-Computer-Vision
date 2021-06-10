import json
import os
import numpy as np
import glob
import PIL.Image as Image
from omegaconf import DictConfig
from tqdm import tqdm
import wget
import tarfile
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import zipfile
import gdown
import os
import PIL.Image as Image
import h5py
import pandas as pd

MIN_SIZE = 20


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


def random_square(img, meta, rng):
    h, w, _ = img.shape
    available_modes = []
    leftest = np.min(np.array(meta['left']))
    if leftest > MIN_SIZE:
        available_modes.append(0)
    rightest = np.max(np.array(meta['left']) + np.array(meta['width']))
    if w - rightest > MIN_SIZE:
        available_modes.append(1)
    topest = np.min(np.array(meta['top']))
    if topest > MIN_SIZE:
        available_modes.append(2)
    lowest = np.max(np.array(meta['top']) + np.array(meta['height']))
    if h - lowest > MIN_SIZE:
        available_modes.append(3)

    mode_boundaries = [
        [0, leftest, 0, h],
        [rightest, w, 0, h],
        [0, w, 0, topest],
        [0, w, lowest, h]
    ]
    if len(available_modes) == 0:
        raise ValueError('no possible part')
    mode = rng.integers(0, len(available_modes))

    l, r, t, b = random_square_from_boundaries(*mode_boundaries[available_modes[mode]], rng)
    return img[t:b, l:r]

def random_square_from_boundaries(l, r, t, b, rng):
    w = r - l
    h = b - t
    size = rng.integers(MIN_SIZE, min(w, h))
    left = l + rng.integers(0, w - size)
    right = left + size
    top = t + rng.integers(0, h - size)
    bottom = top + size
    return int(left), int(right), int(top), int(bottom)

def load_meta(meta_csv):
    df = pd.read_csv(meta_csv)
    df['height'] = df.height.apply(json.loads)
    df['label'] = df.label.apply(json.loads)
    df['left'] = df.left.apply(json.loads)
    df['top'] = df.top.apply(json.loads)
    df['width'] = df.width.apply(json.loads)
    return df


def filter_images(df):
    ok_ind = []
    rejected = []
    rng = np.random.default_rng(12345)
    print('filtering images for no-digit')
    for i in tqdm(range(len(df))):
        try:
            meta = df.iloc[i]
            image = Image.open(os.path.join('./data/train/', meta.file))
            img = np.array(image)
            random_square(img, meta, rng)
            ok_ind.append(i)
        except ValueError as e:
            rejected.append((i, meta.file))
    return ok_ind


class NoDigitDataset(Dataset):
    def __init__(self, transform, folder="./", is_val=False):
        self.is_val = is_val
        self.rng = np.random.default_rng(12345)
        self.transform = transform
        self.folder = folder
        meta_csv = os.path.join(folder, 'meta.csv')
        if os.path.isfile(meta_csv):
            print('loading metadata from cache')
            self.df = load_meta(meta_csv)
        else:
            print('parsing metadata')
            self.df = get_metadata(folder=folder)
            print('caching')
            self.df.to_csv(meta_csv, index=None)

        ind_csv = os.path.join(folder, 'ind.csv')
        if os.path.isfile(ind_csv):
          print('indices from cashe')
          self.filtered_indices = pd.read_csv(ind_csv).ind
        else:
          self.filtered_indices = filter_images(self.df)
          pd.DataFrame({'ind': self.filtered_indices}).to_csv(ind_csv, index=False)
          print('got: ', len(self.filtered_indices))


    def __len__(self):
        # 'Returns the total number of samples'
        return len(self.filtered_indices)  # this could be arbitrarily larger

    def __getitem__(self, idx):
        # 'Generates one sample of data'
        ind = self.filtered_indices[idx]
        if self.is_val:
            rng = np.random.default_rng(ind)
        else:
            rng = self.rng
        meta = self.df.iloc[ind]
        image = Image.open(os.path.join(self.folder, meta.file))
        img = np.array(image)

        X = Image.fromarray(random_square(img, meta, rng))
        X = self.transform(X)
        return X, 10


def move_all_files_in_dir(src_dir, dst_dir):
    # Check if both the are directories
    if os.path.isdir(src_dir) and os.path.isdir(dst_dir) :
        # Iterate over all the files in source directory
        for filePath in glob.glob(src_dir + '/*'):
            # Move each file to destination Directory
            shutil.move(filePath, dst_dir);
    else:
        print("srcDir & dstDir should be Directories")


def get_data_no_digit(size, train_augmentation, batch_size, base_path: str = './'):
    train_transform, valid_transform = get_transforms(size, train_augmentation)

    if not os.path.isfile('train.tar.gz'):
        url = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
        print('Downloading data')
        filename = wget.download(url)
    else:
        filename = 'train.tar.gz'
        print('Using cached train.tar.gz')

    if not os.path.isdir('data/train'):
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

        dir_path = 'data/train'
        try:
            os.mkdir(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))
        move_all_files_in_dir('train', dir_path)
    else:
        print('Using cached data/train')

    train_set = NoDigitDataset(folder='./data/train', transform=train_transform)
    valid_set = NoDigitDataset(folder='./data/train', transform=valid_transform, is_val=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader


def get_data_svhn(size, train_augmentation, batch_size, base_path: str = './'):
    train_transform, valid_transform = get_transforms(size, train_augmentation)
    train_set = datasets.SVHN('./data', split='train', download=True, transform=train_transform)
    valid_set = datasets.SVHN('./data', split='test', download=True, transform=valid_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader


def get_transforms(size, train_augmentation):
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = list()
    if 'random_crop' in train_augmentation:
        train_transform.append(transforms.Resize((int(1.1 * size), int(1.1 * size))))
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


def plot_data(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(20, 10))

    for i in range(21):
        plt.subplot(5, 7, i + 1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        # plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')
