'''ResNet in PyTorch.
Code pulled from https://github.com/kuangliu/pytorch-cifar
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, dropout=0.05):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(p=self.dropout)

    def forward(self, x):
        out = F.relu(self.dropout(self.bn(self.conv(x))))
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.05):
        super(TransitionBlock, self).__init__()
        self.dropout = dropout
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(p=self.dropout)

    def forward(self, x):
        x = F.relu(self.dropout(self.bn(self.max_pool(self.conv(x)))))
        return x

class LayeredBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.05):
        super(LayeredBlock, self).__init__()
        self.dropout = dropout
        self.transition_block = TransitionBlock(in_planes, planes, stride, dropout)
        self.convb1 = ConvBlock(planes, planes, stride, dropout)
        self.convb2 = ConvBlock(planes, planes, stride, dropout)

    def forward(self, x):
        x = self.transition_block(x)
        r = self.convb2(self.convb1(x))
        out = x + r

        return out


class CustomResNet(nn.Module):
    def __init__(self, dropout = 0.0, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv = ConvBlock(3, 64, 1, dropout)
        self.layer1 = LayeredBlock(64, 128, 1, dropout)
        self.layer2 = TransitionBlock(128, 256, 1, dropout)
        self.layer3 = LayeredBlock(256, 512, 1, dropout)
        self.max_pool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out)
def cusResNet(dropout = 0.05, num_classes=10):
    return CustomResNet(dropout, num_classes)


def test():
    net = custom_ResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())