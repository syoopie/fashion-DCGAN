import torch
from torch import nn
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(self):
        super().__init__()
        # receive a vector of 100 random numbers
        self.dense0 = nn.Linear(100, 32)
        self.dense1 = nn.Linear(32, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 256)
        self.dense4 = nn.Linear(256, 784)
        self.dropout = nn.Dropout(0.3)

    def forward(self, X):
        # leaky_relu (value * factor)
        X = self.dropout(F.leaky_relu(self.dense0(X)))
        X = self.dropout(F.leaky_relu(self.dense1(X)))
        X = self.dropout(F.leaky_relu(self.dense2(X)))
        X = self.dropout(F.leaky_relu(self.dense3(X)))
        X = torch.tanh(self.dense4(X))  # literature
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
        X = self.dropout(F.leaky_relu(self.dense0(X)))
        X = self.dropout(F.leaky_relu(self.dense1(X)))
        X = self.dropout(F.leaky_relu(self.dense2(X)))
        X = self.dense3(X)
        return X

# More layers
