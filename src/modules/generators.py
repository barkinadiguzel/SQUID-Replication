import torch.nn as nn

class TeacherGenerator(nn.Module):
    def __init__(self, embed_dim, output_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, output_channels, 1)
        )

    def forward(self, x):
        return self.conv(x)

class StudentGenerator(nn.Module):
    def __init__(self, embed_dim, output_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, output_channels, 1)
        )

    def forward(self, x):
        return self.conv(x)
