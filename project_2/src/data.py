from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import requests
import os
import zipfile
import shutil


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
    os.rmdir(os.path.join(directory_to_extract_to, 'LIDC_crops'))


def get_dataset(url, data_path):
    raw_file, raw_folder = os.path.join(data_path, 'raw.zip'), os.path.join(data_path, 'raw')

    if not os.path.isfile(raw_file):
        print('Downloading data')
        download_url(url, raw_file)
    else:
        print(f'Data already downloaded at {raw_file}')

    if not os.path.isdir(raw_file):
        print('Extracting data')
        extract_data(raw_file, raw_folder)
    else:
        print(f'Data already extracted at {raw_folder}')

    train_set, valid_set, test_set = None, None, None
    return train_set, valid_set, test_set


class LIDCIDRIDataset(Dataset):
    def __init__(self, transform, url):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def get_dataloaders(size, batch_size, url, data_path):
    train_set, valid_set, test_set = get_dataset(url, data_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader


# def get_transforms(size, train_augmentation):
#     norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
#     train_transform = list()
#     if 'random_crop' in train_augmentation:
#         train_transform.append(transforms.Resize((int(1.1 * size), int(1.1 * size))))
#         train_transform.append(transforms.RandomCrop((size, size)))
#     else:
#         train_transform.append(transforms.Resize((size, size)))
#     if 'random_horizontal_flip' in train_augmentation:
#         train_transform.append(transforms.RandomHorizontalFlip())
#     if 'color_jitter' in train_augmentation:
#         train_transform.append(transforms.ColorJitter())
#     train_transform.append(transforms.ToTensor())
#     train_transform.append(transforms.Normalize(norm_mean, norm_std))
#     train_transform = transforms.Compose(train_transform)
#
#     valid_transform = [transforms.Resize((size, size)),
#                        transforms.ToTensor(),
#                        transforms.Normalize(norm_mean, norm_std)]
#     valid_transform = transforms.Compose(valid_transform)
#     return train_transform, valid_transform

