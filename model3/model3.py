import torch
from torch import nn
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(100, 512)
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.batchnorm1 = nn.BatchNorm1d(512, momentum=0.8)
        self.fc2 = nn.Linear(512, 256)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.batchnorm2 = nn.BatchNorm1d(256, momentum=0.8)
        self.fc3 = nn.Linear(256, 128)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.batchnorm3 = nn.BatchNorm1d(128, momentum=0.8)
        self.fc4 = nn.Linear(128, 784)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyrelu1(x)
        x = self.batchnorm1(x)
        x = self.fc2(x)
        x = self.leakyrelu2(x)
        x = self.batchnorm2(x)
        x = self.fc3(x)
        x = self.leakyrelu3(x)
        x = self.batchnorm3(x)
        x = self.fc4(x)
        x = x.view(x.shape[0], 28, 28)

        return x


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = nn.Linear(784, 256)
        self.leakyrelu0 = nn.LeakyReLU(0.2)
        self.dense1 = nn.Linear(256, 128)
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(128, 64)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.dense3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.shape[0], 28 * 28)
        x = self.dropout(self.leakyrelu0(self.dense0(x)))
        x = self.dropout(self.leakyrelu1(self.dense1(x)))
        x = self.dropout(self.leakyrelu2(self.dense2(x)))
        x = self.dense3(x)
        x = self.sigmoid(x)
        return x


# More fine tuned linear models