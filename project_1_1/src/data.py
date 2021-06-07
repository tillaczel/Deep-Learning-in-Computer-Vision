from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision import datasets
from torchvision import transforms

# TODO: change all CIFAR calls to hotdogs
class HotDogDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform_train = transforms.Compose([
            # TODO: data augmentation
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.dims = (3, 32, 32)

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            data_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            # TODO: change split sizes for hotdogs
            self.data_train, self.data_val = random_split(data_full, [40000, 10000])

        if stage == 'test' or stage is None:
            self.data_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform_test)


    def (self):
        # TODO: batch size?
        return DataLoader(self.data_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=32)