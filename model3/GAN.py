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
train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

G = generator()
D = discriminator()

G_optimizer = optim.Adam(G.parameters(), lr=0.002)
D_optimizer = optim.Adam(D.parameters(), lr=0.002)

criterion = nn.BCEWithLogitsLoss()  # combine sigmoid with and BCELoss

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU or CPU:", device)

G.to(device)
D.to(device)

G.train()
D.train()

for epoch in range(50):
    D_running_loss = 0
    G_running_loss = 0

    for i, (images_real, _) in enumerate(train_loader):
        batch_size = images_real.size(0)
        images_real = images_real * 2 - 1
        images_real = images_real.to(device)

        # Train The Generator
        G_optimizer.zero_grad()
        random = np.random.uniform(low=-1., high=1., size=(batch_size, 100))
        random = torch.from_numpy(random).float().to(device)
        images_false = G.forward(random)
        output_false = D.forward(images_false)
        labels_false = torch.ones(batch_size).to(device)
        G_loss = criterion(output_false.view(
            *labels_false.shape), labels_false)
        G_loss.backward()
        G_optimizer.step()

        # Train The Discriminator
        D_optimizer.zero_grad()
        outputs_real = D.forward(images_real)
        labels_real = (torch.ones(batch_size) * 0.9).to(device)
        D_loss_real = criterion(outputs_real.view(
            *labels_real.shape), labels_real)

        random = np.random.uniform(-1., 1., (batch_size, 100))
        random = torch.from_numpy(random).float().to(device)
        images_false = G.forward(random)
        outputs_false = D.forward(images_false)
        labels_false = torch.zeros(batch_size).to(device)
        D_loss_false = criterion(outputs_false.view(
            *labels_false.shape), labels_false)

        D_loss = D_loss_real + D_loss_false
        D_loss.backward()
        D_optimizer.step()

        D_running_loss += D_loss.item()
        G_running_loss += G_loss.item()

    # print the loss after each epoch
    D_running_loss /= len(train_loader)
    G_running_loss /= len(train_loader)
    print('EPOCH {:03d} finalized: discriminator loss {:03.6f} - gererator loss {:03.6f}'.format(
        epoch + 1, D_running_loss, G_running_loss))

    # fig, ax = plt.subplots(1, 5, figsize=(10, 5))
    # for i in range(5):
    #     ax[i].imshow(images_false.cpu().detach().numpy()
    #                  [i].reshape(28, 28), cmap='gray')
    #     ax[i].xaxis.set_visible(False)
    #     ax[i].yaxis.set_visible(False)
    # plt.show()


torch.save(G.state_dict(), 'generator.pt')
torch.save(D.state_dict(), 'discriminator.pt')
