from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import requests
import os
import zipfile
import shutil
from PIL import Image
import torch


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def extract_data(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    move_from = os.path.join(directory_to_extract_to, 'LIDC_crops', 'LIDC_DLCV_version')
    set_folders = os.listdir(move_from)
    for set_folder in set_folders:
        shutil.move(os.path.join(move_from, set_folder), os.path.join(directory_to_extract_to, set_folder))
    shutil.rmtree(os.path.join(directory_to_extract_to, 'LIDC_crops'))


def get_dataset(url, data_path, train_transform, valid_transform, seg_transform, seg_reduce):
    os.mkdir(data_path) if not os.path.isdir(data_path) else None
    raw_file, raw_folder = os.path.join(data_path, 'raw.zip'), os.path.join(data_path, 'raw')

    if not os.path.isfile(raw_file):
        print('Downloading data')
        download_url(url, raw_file)
    else:
        print(f'Data already downloaded at {raw_file}')

    if not os.path.isdir(raw_folder):
        print('Extracting data')
        extract_data(raw_file, raw_folder)
    else:
        print(f'Data already extracted at {raw_folder}')

    train_set = LIDCIDRIDataset(os.path.join(raw_folder, 'train'), train_transform, seg_transform, seg_reduce=seg_reduce)
    valid_set = LIDCIDRIDataset(os.path.join(raw_folder, 'val'), valid_transform, seg_transform, seg_reduce=seg_reduce)
    test_set = LIDCIDRIDataset(os.path.join(raw_folder, 'test'), valid_transform, seg_transform, seg_reduce=seg_reduce)
    return train_set, valid_set, test_set


class LIDCIDRIDataset(Dataset):
    def __init__(self, folder_path, img_transform, seg_transform, seg_reduce='mean'):
        self.img_path = os.path.join(folder_path, 'images')
        self.seg_path = os.path.join(folder_path, 'lesions')

        self.idx2fname = {i: fname for i, fname in enumerate(os.listdir(self.img_path))}

        self.img_transform, self.seg_transform = img_transform, seg_transform
        self.seg_reduce = seg_reduce
        self.get_seg = self._init_get_seg()

    def __len__(self):
        return len(self.idx2fname)

    def __getitem__(self, idx):
        fname = self.idx2fname[idx]
        img = self.img_transform(Image.open(os.path.join(self.img_path, fname)))
        seg = self.get_seg(fname)
        return img, seg

    def _init_get_seg(self):
        if self.seg_reduce == 'mean':
            return lambda fname: torch.mean(self._read_seg(fname, range(4)), dim=0)
        elif self.seg_reduce == 'all':
            return lambda fname: self._read_seg(fname, range(4))
        elif type(self.seg_reduce) == int:
            return lambda fname: self._read_seg(fname, [self.seg_reduce])[0]
        else:
            raise ValueError(f'{self.seg_reduce} is not valid segmentation reduction')

    def _read_seg(self, fname, idxs):
        segs = [Image.open(os.path.join(self.seg_path, f'{fname[:-4]}_l{idx}.png')) for idx in idxs]
        return torch.stack(list(map(self.seg_transform, segs)))


def get_dataloaders(size, train_augmentation, batch_size, url, data_path, seg_reduce):
    train_transform, valid_transform, seg_transform = get_transforms(size, train_augmentation)
    train_set, valid_set, test_set = get_dataset(url, data_path, train_transform, valid_transform, seg_transform, seg_reduce)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


def get_transforms(size, train_augmentation):
    train_transform = list()
    if 'random_crop' in train_augmentation:
        train_transform.append(transforms.Resize((int(1.1 * size), int(1.1 * size))))
        train_transform.append(transforms.RandomCrop((size, size)))
    else:
        train_transform.append(transforms.Resize((size, size)))
    if 'random_horizontal_flip' in train_augmentation:
        train_transform.append(transforms.RandomHorizontalFlip())
    if 'random_vertical_flip' in train_augmentation:
        train_transform.append(transforms.RandomVerticalFlip())
    if 'random_rotation' in train_augmentation:
        train_transform.append(transforms.RandomRotation(10))
    if 'color_jitter' in train_augmentation:
        train_transform.append(transforms.ColorJitter())
    train_transform.append(transforms.ToTensor())
    train_transform = transforms.Compose(train_transform)

    valid_transform = [transforms.Resize((size, size)), transforms.ToTensor()]
    valid_transform = transforms.Compose(valid_transform)

    seg_transform = [transforms.Resize((size, size)), transforms.ToTensor()]
    seg_transform = transforms.Compose(seg_transform)
    return train_transform, valid_transform, seg_transform

