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


def download_data():
    if not os.path.exists('./hotdog_nothotdog'):
        url = 'https://drive.google.com/uc?id=1hwyBl4Fa0IHihun29ahszf1M2cxn9TFk'
        gdown.download(url, './hotdog_nothotdog.zip', quiet=False)

        with zipfile.ZipFile('./hotdog_nothotdog.zip', 'r') as zip_ref:
            zip_ref.extractall('/dev/null')


class Hotdog_NotHotdog(Dataset):
    def __init__(self, train, transform, data_path='hotdog_nothotdog'):
        # 'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        # 'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def get_data(size, batch_size):
    train_transform = transforms.Compose([transforms.Resize((size, size)),
                                          transforms.ToTensor()])
    valid_transform = transforms.Compose([transforms.Resize((size, size)),
                                          transforms.ToTensor()])

    train_set = Hotdog_NotHotdog(train=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_set = Hotdog_NotHotdog(train=False, transform=valid_transform)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader


def plot_data(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(20,10))
    
    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')
