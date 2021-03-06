import json
import numpy as np
import glob
from tqdm import tqdm
import wget
import tarfile
import shutil

import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import h5py
import pandas as pd

MIN_SIZE = 32


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


def random_square(img, meta, rng, overlap=0.2):
    h, w, _ = img.shape
    available_modes = []
    probs = []
    leftest = int(np.min(np.array(meta['left']) + overlap * np.array(meta['width'])))
    if leftest > MIN_SIZE:
        available_modes.append(0)
        probs.append(leftest)
    rightest = int(np.max(np.array(meta['left']) + (1- overlap) * np.array(meta['width'])))
    if w - rightest > MIN_SIZE:
        available_modes.append(1)
        probs.append(w - rightest)
    topest = int(np.max(np.array(meta['top']) + overlap * np.array(meta['height'])))
    if topest > MIN_SIZE:
        available_modes.append(2)
        probs.append(topest)
    lowest = int(np.max(np.array(meta['top']) + (1 - overlap) * np.array(meta['height'])))
    if h - lowest > MIN_SIZE:
        available_modes.append(3)
        probs.append(h - lowest)

    if len(available_modes) == 0:
        raise ValueError('no possible part')

    probs = np.array(probs)
    probs = probs / probs.sum() # normalize

    mode_boundaries = [
        [0, leftest, 0, h],
        [rightest, w, 0, h],
        [0, w, 0, topest],
        [0, w, lowest, h]
    ]
    mode = rng.choice(available_modes, p=probs) # get bigger

    l, r, t, b = random_square_from_boundaries(*mode_boundaries[mode], rng)
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


def filter_images(df, overlap=0.2):
    ok_ind = []
    rejected = []
    rng = np.random.default_rng(12345)
    print('filtering images for no-digit')
    for i in tqdm(range(len(df))):
        try:
            meta = df.iloc[i]
            image = Image.open(os.path.join('./data/train/', meta.file))
            img = np.array(image)
            random_square(img, meta, rng, overlap=overlap)
            ok_ind.append(i)
        except ValueError as e:
            rejected.append((i, meta.file))
    return ok_ind


class NoDigitDataset(Dataset):
    def __init__(self, transform, folder="./", is_val=False, overlap=0.2):
        self.is_val = is_val
        self.overlap = overlap
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

        ind_csv = os.path.join(folder, f'ind_{self.overlap:.3f}.csv')
        if os.path.isfile(ind_csv):
          print('indices from cashe')
          self.filtered_indices = pd.read_csv(ind_csv).ind
        else:
          self.filtered_indices = filter_images(self.df, overlap=self.overlap)
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

        X = Image.fromarray(random_square(img, meta, rng, overlap=self.overlap))
        X = self.transform(X)
        return X, 10

class RawDataset(Dataset):
    def __init__(self, transform, resize_ratios=(0.5, 0.7, 0.9), folder="./"):
        self.size = 224 # hardcoded
        self.resize_heights = (self.size / np.array(list(resize_ratios))).astype(int)
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

    def __len__(self):
        # 'Returns the total number of samples'
        return len(self.df)  # this could be arbitrarily larger

    def __getitem__(self, idx):
        # 'Generates one sample of data'
        meta = self.df.iloc[idx]
        original_image = Image.open(os.path.join(self.folder, meta.file))
        w, h = original_image.size
        resized = [] # list
        ratios = []
        for height in self.resize_heights:
            ratio = height/h
            width = int(w * ratio)
            X = original_image.resize((width, height))
            X = self.transform(X)
            ratios.append(ratio)
            resized.append(X)
        # list of resized images, original,
        # ratios used for resizing and original metadata
        return resized, original_image, ratios, meta


def move_all_files_in_dir(src_dir, dst_dir):
    # Check if both the are directories
    if os.path.isdir(src_dir) and os.path.isdir(dst_dir) :
        # Iterate over all the files in source directory
        for filePath in glob.glob(src_dir + '/*'):
            # Move each file to destination Directory
            shutil.move(filePath, dst_dir);
    else:
        print("srcDir & dstDir should be Directories")

def download_if_necessary(split):
    assert split in ['train', 'test', 'extra']
    if not os.path.isdir(f'data/{split}'):
        if not os.path.isfile('f{split}.tar.gz'):
            url = f'http://ufldl.stanford.edu/housenumbers/{split}.tar.gz'
            print('Downloading data')
            filename = wget.download(url)
        else:
            filename = f'{split}.tar.gz'
            print(f'Using cached {split}.tar.gz')

        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

        if not os.path.isdir('data'):
            os.mkdir('data')
        dir_path = f'data/{split}'
        try:
            os.mkdir(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))
        move_all_files_in_dir(f'{split}', dir_path)
    else:
        print(f'Using cached data/{split}')

def get_data_raw(split='test', resize_ratios=(0.5, 0.7, 0.9)):
    download_if_necessary(split)
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return RawDataset(transform, resize_ratios=resize_ratios, folder=f"./data/{split}")

def get_data_no_digit(size, train_augmentation, overlap):
    train_transform, valid_transform = get_transforms(size, train_augmentation)
    download_if_necessary('train')
    train_set = NoDigitDataset(folder='./data/train', transform=train_transform, overlap=overlap)
    valid_set = NoDigitDataset(folder='./data/train', transform=valid_transform, is_val=True, overlap=overlap)
    return train_set, valid_set


def get_data_svhn(size, train_augmentation):
    train_transform, valid_transform = get_transforms(size, train_augmentation)
    train_set = datasets.SVHN('./data/svhn', split='train', download=True, transform=train_transform)
    valid_set = datasets.SVHN('./data/svhn', split='test', download=True, transform=valid_transform)
    return train_set, valid_set


def get_dataloaders(size, train_augmentation, batch_size, overlap=0.2):
    train_set_no_digit, valid_set_no_digit = get_data_no_digit(size, train_augmentation, overlap=overlap)
    train_set_svhn, valid_set_svhn = get_data_svhn(size, train_augmentation)
    train_set, valid_set = ConcatDataset([train_set_no_digit, train_set_svhn]), ConcatDataset([valid_set_no_digit, valid_set_svhn])
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
