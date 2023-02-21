'''ResNet in PyTorch.
Code pulled from https://github.com/kuangliu/pytorch-cifar
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN model definition
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, dropout=0.1):
        super(ConvNet, self).__init__()
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # input: 3*32*32, output: 16*32*32, RF: 3*3
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  # input: 16*32*32, output: 32*32*32, RF: 5*5
            nn.BatchNorm2d(32),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),  # input: 32*32*32, output: 48*32*32, RF: 7*7
            nn.BatchNorm2d(48),
            nn.Dropout(self.dropout)
        )

        self.GAP = nn.Sequential(
            nn.AvgPool2d(32, 32),  # input: 16*7*7, output: 16*3*3, RF: 38*38
        )

    def forward(self, x):
        x = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))

        x = self.GAP(x)
        x = x.view(x.size(0), -1)

        return x


class Ultimus(nn.Module):

    def __init__(self, in_nodes, d_k, dropout):
        super(Ultimus, self).__init__()

        self.in_nodes = in_nodes
        self.d_k = d_k
        self.dropout = dropout

        self.query = nn.Linear(self.in_nodes, self.d_k)
        self.keys = nn.Linear(self.in_nodes, self.d_k)
        self.values = nn.Linear(self.in_nodes, self.d_k)
        self.fc1 = nn.Linear(self.d_k, self.in_nodes)

    def forward(self, x):
        Q = F.relu(self.query(x))
        V = F.relu(self.values(x))
        K = F.relu(self.keys(x))

        SA = torch.mm(torch.transpose(Q, 0, 1), K) / math.sqrt(self.d_k)
        SA_soft = F.softmax(SA, dim=1)
        ATT = torch.mm(V, SA_soft)
        out = F.relu(self.fc1(ATT))

        return out


class AttentionNet(nn.Module):
    def __init__(self, in_nodes, d_k, dropout=0.1, no_classes=10):
        super(AttentionNet, self).__init__()

        self.in_nodes = in_nodes
        self.d_k = d_k
        self.no_classes = no_classes
        self.dropout = dropout

        self.conv = ConvNet(dropout=self.dropout)

        self.ultimus = Ultimus(self.in_nodes, self.d_k, self.dropout)

        self.fc = nn.Linear(self.in_nodes, self.no_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.ultimus(x)
        x = self.ultimus(x)
        x = self.ultimus(x)
        x = self.ultimus(x)

        x = self.fc(x)

        # return x
        return F.softmax(x, dim=1)


def AttnNet(in_nodes, d_k, dropout=0.1, no_classes=10):
    return AttentionNet(in_nodes, d_k, dropout, no_classes)


def test():
    net = AttnNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())