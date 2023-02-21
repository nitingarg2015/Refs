from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision
from torchsummary import summary
import numpy as np
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.optim import Adam
from torch.optim import SGD


def train(model, device, train_loader, optimizer, epoch, criterion, scheduler, lr_trend, lambda_l1=0):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)

        if (lambda_l1 > 0):
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1

        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # updating LR
        if scheduler:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                lr_trend.append(scheduler.get_last_lr()[0])
        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_acc = 100 * correct / len(train_loader.dataset)

    print(
        f'\nAverage Training Loss={train_loss / len(train_loader.dataset)}, Accuracy={100 * correct / len(train_loader.dataset)}')

    return train_loss, train_acc


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)

    return test_loss, test_acc


def fit_model(model, optimizer, criterion, train_loader, test_loader, EPOCHS, device, lambda_l1=0, scheduler=None):
    train_losses = []
    train_acc_all = []
    test_losses = []
    test_acc_all = []

    lr_trend = []

    for epoch in range(EPOCHS):
        print("EPOCH: {} (LR: {})".format(epoch + 1, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(model, device, trainloader, optimizer, epoch, criterion, scheduler, lr_trend,
                                      lambda_l1)

        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss / len(train_loader.dataset))
        train_acc_all.append(train_acc)
        test_losses.append(test_loss)
        test_acc_all.append(test_acc)

    return model, train_losses, train_acc_all, test_losses, test_acc_all