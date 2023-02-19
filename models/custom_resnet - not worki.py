'''ResNet in PyTorch.
Code pulled from https://github.com/kuangliu/pytorch-cifar
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayeredBlock(nn.Module):

    def __init__(self, in_planes, planes, layer_no):
        super(LayeredBlock, self).__init__()

        self.layer_no = layer_no

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        if self.layer_no % 2 == 1:

            x = F.relu(self.bn1(self.maxpool(self.conv1(x))))
            out = F.relu(self.bn2(self.conv2(x)))
            out = F.relu(self.bn3(self.conv3(out)))
            out += x

        else:

            out = F.relu(self.bn1(self.maxpool(self.conv1(x))))

        return out


class custom_ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(custom_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = LayeredBlock(in_planes=64, planes=128, layer_no=1)
        self.layer2 = LayeredBlock(in_planes=128, planes=256, layer_no=2)
        self.layer3 = LayeredBlock(in_planes=256, planes=512, layer_no=3)

        self.maxpool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out)


def cusResNet():
    return custom_ResNet()


def test():
    net = custom_ResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())