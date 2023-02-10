'''
Contains following functions:
Define classes for CIFAR train and test datasets
load_CIFAR - for loading CIFAR dataset
get_miss_classified - to retrieve miss classified images
getFractionsMissed - to retried fractions missed by each class in the dataset
plot_LossAndAcc - plot train/test loss and accuracies
get_mean_and_std - returns mean and std of the dataset
'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import torchvision

class TrainSet(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=True, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

class TestSet(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=False, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def load_CIFAR(transform, train=True, batch_size=128):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if train == True:

        dataset = TrainSet(transform=transform)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = TrainSet(transform=transform)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return classes, data_loader

### Functions to plot misclassified images

'''
function to return data points for plotting misses
It accepts the following inputs:
1. trained model - trained on GPU, 2. device - cuda device, 3. images as a tensor (channel*shape*shape),
4) labels as 1D torch tensor

Outputs provided for all incorrect predictions, where length of each list = number of batches:
1. data_images - List of torch tensors of images, if no mispredictions in a given batch then empty tensor
2. pred_labels - List of Predicted labels, if no mispredictions in a given batch then empty tensor
3. target_labels - Ground truth or target labels, if no mispredictions in a given batch then empty tensor
'''
import torch
import collections

def get_miss_classified(model, device, images, labels):
    #images to labels to GPU
    if not (images.is_cuda):
        images = images.to(device)
    if not (labels.is_cuda):
        labels = labels.to(device)

    model.eval()

    with torch.no_grad():
        output = model(images)
        #predict labels
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #identify positions that are misclassified
        misses = torch.where(torch.not_equal(pred.squeeze(), labels))[0].cpu()

        # get missed information convert tensors back to cpu() for plotting images
        if images.is_cuda:
            data_images = images[misses].cpu()
        if labels.is_cuda:
            target_labels = labels[misses].cpu()
        if pred.is_cuda:
            pred_labels = pred[misses].cpu()

    return data_images, target_labels, pred_labels

'''
Function getFractionsMissed --> 
Input: model, device, test_loader
Output: fractions missed by each class
'''
def getFractionsMissed(model, device, test_loader):
    model.eval()
    missed_targets = []  # contains list of target labels by batch
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index

            misses = torch.where(torch.not_equal(pred.squeeze(), target))
            missed_targets.append(target[misses].cpu())
            targets.append(target.cpu())

    targets = [x.item() for item in targets for x in item]
    missed_targets = [x.item() for item in missed_targets for x in item]

    # using Counter to find frequency of elements
    target_freq = collections.Counter(targets)
    miss_freq = collections.Counter(missed_targets)

    # calculate fraction misses for each class
    fractions_dict = {key: miss_freq[key] / target_freq.get(key, 0)
                      for key in target_freq.keys()}
    # sort fractions_dict by keys
    keys = list(fractions_dict.keys())
    keys.sort()
    fractions_dict = {i: fractions_dict[i] for i in keys}

    return fractions_dict

def plot_LossAndAcc(train_acc,train_losses,test_acc,test_losses):

    # function for plotting test and training loss/ accuracy
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    axs[0].plot(train_losses, label='Training Losses')
    axs[0].plot(test_losses, label='Test Losses')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title("Loss")

    axs[1].plot(train_acc, label='Training Accuracy')
    axs[1].plot(test_acc, label='Test Accuracy')
    axs[1].legend(loc='lower right')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title("Accuracy")

    plt.show()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
