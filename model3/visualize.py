from torchvision import datasets, transforms
import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from models import generator, discriminator

print("Torch Version:", torch.__version__)

torch.manual_seed(42)

class FashionDataset(Dataset):
    def __init__(self, transform=None):
        self.train_data = datasets.FashionMNIST(
            root='data',
            train=True,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]


transform = transforms.ToTensor()
train = FashionDataset(transform=transform)

# Define the number of images you want to display
num_images = 10

# Plot the images
fig, ax = plt.subplots(1, num_images, figsize=(10, 5))
for i in range(num_images):
    ax[i].imshow(train[i][0].squeeze(), cmap='gray')
    ax[i].xaxis.set_visible(False)
    ax[i].yaxis.set_visible(False)
plt.show()
