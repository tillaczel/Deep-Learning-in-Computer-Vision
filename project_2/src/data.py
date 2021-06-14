from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import requests


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


class LIDCIDRIDataset(Dataset):
    def __init__(self, transform, url):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def get_dataset(url, data_path):
    download_url(url, data_path)
    return None, None


def get_dataloaders(size, batch_size, url, data_path):
    train_set, valid_set = get_dataset(url, data_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
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

