from torch import nn

class generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(100, 7 * 7 * 128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
