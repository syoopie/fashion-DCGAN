from torchvision import datasets, transforms
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import generator, discriminator

print("Torch Version:", torch.__version__)
torch.manual_seed(42)


train_data = pd.read_csv('./data/train.csv')
X_train = train_data.drop('label', axis=1)
X_train = X_train.values
X_train = X_train.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_train = X_train / 255
X_train = X_train * 2 - 1.

G = generator()
D = discriminator()

# Define the optimizers for each model
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# Define the loss function
criterion = nn.BCELoss()

# Set up training parameters
num_epochs = 200
batch_size = 100
noise_shape = 100

# Create device object to move tensors to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the device
G.to(device)
D.to(device)

# Set discriminator's trainable flag to False
D.trainable = False

# Train the GAN
for epoch in range(num_epochs):
    print(f"Currently on Epoch {epoch+1}")

    for i in range(X_train.shape[0] // batch_size):

        if (i + 1) % 50 == 0:
            print(
                f"\tCurrently on batch number {i + 1} of {X_train.shape[0]//batch_size}")

        # Generate fake images
        noise = torch.randn(batch_size, noise_shape, device=device)
        gen_images = G(noise)

        # Get real images
        real_images = X_train[i * batch_size:(i + 1) * batch_size]
        real_images = torch.tensor(
            real_images, dtype=torch.float32, device=device)

        # Train the discriminator on real images
        D.train()
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, device=device)
        real_preds = D(real_images).view(-1)
        d_loss_real = criterion(real_preds, real_labels)
        d_loss_real.backward()

        # Train the discriminator on fake images
        fake_labels = torch.zeros(batch_size, device=device)
        fake_preds = D(gen_images.detach()).view(-1)
        d_loss_fake = criterion(fake_preds, fake_labels)
        d_loss_fake.backward()
        optimizer_D.step()

        # Train the generator
        G.train()
        optimizer_G.zero_grad()
        gen_labels = torch.ones(batch_size, device=device)
        gen_preds = D(gen_images).view(-1)
        d_g_loss_batch = criterion(gen_preds, gen_labels)
        d_g_loss_batch.backward()
        optimizer_G.step()

    # Plot generated images
    # if epoch % 10 == 0:
    #     G.eval()
    #     samples = 10
    #     noise = torch.randn(samples, noise_shape, device=device)
    #     x_fake = G(noise).detach().cpu().numpy()

    #     for k in range(samples):
    #         plt.subplot(2, 5, k + 1)
    #         plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
    #         plt.xticks([])
    #         plt.yticks([])

    #     plt.tight_layout()
    #     plt.show()
    
    if epoch % 20 == 0:
        print("Save no:", epoch)
        torch.save(G.state_dict(), 'generator.pt')
        torch.save(D.state_dict(), 'discriminator.pt')
        

print('Training is complete')


# Save models
print("Final Save")
torch.save(G.state_dict(), 'generator.pt')
torch.save(D.state_dict(), 'discriminator.pt')

# Generate images after training
G.eval()
samples = 10
noise = torch.randn(samples, noise_shape, device=device)
x_fake = G(noise).detach().cpu().numpy()

for k in range(samples):
    plt.subplot(2, 5, k + 1)
    plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()
