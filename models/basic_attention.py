'''
Ultimus in PyTorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvNet(nn.Module):
    def __init__(self, dropout=0.1):
        super(ConvNet, self).__init__()
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride = 2, padding=1),  # input: 3*32*32, output: 16*16*16, RF: 3*3
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride = 2, padding=1),  # input: 16*16*16, output: 32*8*8, RF: 7*7
            nn.BatchNorm2d(32),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride = 2, padding=1),  # input: 32*8*8, output: 48*4*4, RF: 15*15
            nn.BatchNorm2d(48),
            nn.Dropout(self.dropout)
        )

        self.GAP = nn.Sequential(
            nn.AvgPool2d(4, 4),  # input: 48*4*4, output: 48*1*1, RF: 15*15
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

        SA = torch.mm(torch.transpose(Q, 0, 1), K)
        SA_soft = F.softmax(SA, dim=0)/ math.sqrt(self.d_k)
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

        self.ultimus1 = Ultimus(self.in_nodes, self.d_k, self.dropout)
        self.ultimus2 = Ultimus(self.in_nodes, self.d_k, self.dropout)
        self.ultimus3 = Ultimus(self.in_nodes, self.d_k, self.dropout)
        self.ultimus4 = Ultimus(self.in_nodes, self.d_k, self.dropout)

        self.fc = nn.Linear(self.in_nodes, self.no_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)

        x = self.fc(x)

        # return x
        return F.softmax(x, dim=1)


def AttnNet(in_nodes, d_k, dropout=0.1, no_classes=10):
    return AttentionNet(in_nodes, d_k, dropout, no_classes)


def test():
    net = AttnNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())