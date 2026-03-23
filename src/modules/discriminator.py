import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_dim, base_dim*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_dim*2, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(x.size(0), -1)
