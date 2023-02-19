'''
Define classes for CIFAR train and test datasets
load_CIFAR - for loading CIFAR dataset
'''
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsDataSet(Dataset):
    def __init__(self, image_set, transform):
        self.image_set = image_set
        self.transform = transform

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, index):
        image, label = self.image_set.data[index], self.image_set.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def load_CIFAR_dataloader(train, transform, batch_size=128):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if train == True:

        # download test dataset
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True)
    else:
        # download test dataset
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True)
    if transform == True:
        # define transform
        transform = A.Compose([
            #A.HorizontalFlip(),  # Same with transforms.RandomHorizontalFlip()
            A.PadIfNeeded(min_height=36, min_width=36, value=[127, 127, 127], p=1.0),
            A.HorizontalFlip(p=0.2),
            A.Blur(blur_limit=25, p=0.1),
            A.RandomCrop(32, 32, p=1.0),
            #A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=45),
            A.CoarseDropout(min_holes=1, max_holes=1, min_height=8, max_width=8, min_width=8, max_height=8,
                            fill_value=[127, 127, 127], mask_fill_value=None, p=0.5),
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ToTensorV2()
        ])

        # transform using Albumentations
        dataset = AlbumentationsDataSet(image_set=dataset, transform=transform)

        # generate data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

    else:
        # define transform - normalize only
        transform = A.Compose([
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ToTensorV2()
        ])

        # transform using Albumentations
        dataset = AlbumentationsDataSet(image_set=dataset, transform=transform)

        # generate data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return classes, data_loader