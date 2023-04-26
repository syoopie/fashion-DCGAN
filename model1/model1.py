import torch
from torch import nn
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 100->32->64->128->784
        # receive a vector of 100 numbers random
        self.dense0 = nn.Linear(100, 32)
        self.dense1 = nn.Linear(32, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 784)
        self.dropout = nn.Dropout(0.3)

    def forward(self, X):
        X = self.dropout(F.leaky_relu(self.dense0(X), 0.2)
                         )  # leaky_relu (value * factor)
        X = self.dropout(F.leaky_relu(self.dense1(X), 0.2))
        X = self.dropout(F.leaky_relu(self.dense2(X), 0.2))
        X = torch.tanh(self.dense3(X))  # literature
        X = X.view(X.shape[0], 28, 28)  # convert to image (matrix)
        return X

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense0 = nn.Linear(784, 128)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, X):
        X = X.view(X.shape[0], 28 * 28)
        X = self.dropout(F.leaky_relu(self.dense0(X), 0.2))
        X = self.dropout(F.leaky_relu(self.dense1(X), 0.2))
        X = self.dropout(F.leaky_relu(self.dense2(X), 0.2))
        X = self.dense3(X)
        return X
