from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import requests
import os
import zipfile
import shutil
from PIL import Image
import torch
import gdown

def download_url(url, save_path, chunk_size=128):
    gdown.download(url, save_path, quiet=False)
    # r = requests.get(url, stream=True)
    # with open(save_path, 'wb') as fd:
    #     for chunk in r.iter_content(chunk_size=chunk_size):
    #         fd.write(chunk)


def extract_data(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    move_from = os.path.join(directory_to_extract_to, 'horse2zebra')
    set_folders = os.listdir(move_from)
    for set_folder in set_folders:
        shutil.move(os.path.join(move_from, set_folder), os.path.join(directory_to_extract_to, set_folder))
    shutil.rmtree(os.path.join(directory_to_extract_to, 'horse2zebra'))


def get_dataset(url, data_path, train_transform, valid_transform):
    os.mkdir(data_path) if not os.path.isdir(data_path) else None
    raw_file, raw_folder = os.path.join(data_path, 'horse2zebra.zip'), os.path.join(data_path, 'horse2zebra')

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

    train_horse = ImageDataset(os.path.join(raw_folder, 'train', 'A'), train_transform)
    train_zebra = ImageDataset(os.path.join(raw_folder, 'train', 'B'), train_transform)
    test_horse = ImageDataset(os.path.join(raw_folder, 'test', 'A'), valid_transform)
    test_zebra = ImageDataset(os.path.join(raw_folder, 'test', 'B'), valid_transform)
    # valid_set = # TODO: split
    return train_horse, train_zebra, test_horse, test_zebra


class ImageDataset(Dataset):
    def __init__(self, folder_path, img_transform):
        self.img_path = folder_path
        self.fnames = list(os.listdir(self.img_path))
        self.img_transform = img_transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = self.img_transform(Image.open(os.path.join(self.img_path, fname)))
        return img

def get_dataloaders(size, train_augmentation, batch_size, url, data_path):
    train_transform, valid_transform = get_transforms(size, [])
    train_set, valid_set, test_set = get_dataset(url, data_path, train_transform, valid_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


def get_transforms(size, train_augmentation):
    train_transform = list()
    # if 'random_crop' in train_augmentation:
    #     train_transform.append(transforms.Resize((int(1.1 * size), int(1.1 * size))))
    #     train_transform.append(transforms.RandomCrop((size, size)))
    # else:
    #     train_transform.append(transforms.Resize((size, size)))
    # if 'random_horizontal_flip' in train_augmentation:
    #     train_transform.append(transforms.RandomHorizontalFlip())
    # if 'random_vertical_flip' in train_augmentation:
    #     train_transform.append(transforms.RandomVerticalFlip())
    # if 'random_rotation' in train_augmentation:
    #     train_transform.append(transforms.RandomRotation(10))
    # if 'color_jitter' in train_augmentation:
    #     train_transform.append(transforms.ColorJitter())
    train_transform.append(transforms.ToTensor())
    train_transform = transforms.Compose(train_transform)

    valid_transform = [transforms.Resize((size, size)), transforms.ToTensor()]
    valid_transform = transforms.Compose(valid_transform)

    return train_transform, valid_transform

