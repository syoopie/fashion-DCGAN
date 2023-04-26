from torchvision import datasets, transforms
import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch.utils.data import Dataset
from models import generator, discriminator


# Load the generator and discriminator models
G_loaded = generator()
D_loaded = discriminator()

G_loaded.load_state_dict(torch.load('generator.pt'))
D_loaded.load_state_dict(torch.load('discriminator.pt'))


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU or CPU:", device)

random = np.random.uniform(-1., 1., (20, 100))
random = torch.from_numpy(random).float().to(device)

G_loaded.eval()
output = G_loaded.forward(random)

output = output.cpu().detach().numpy()


# Create a 4x5 grid of subplots
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))
# Plot each image in a separate subplot
for i, ax in enumerate(axes.flat):
    ax.imshow(output[i, :].squeeze(), cmap='gray')
    ax.set_axis_off()

# Show the plot
plt.show()
