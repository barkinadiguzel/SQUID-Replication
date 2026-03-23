import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        B, C, H, W = x.size()
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        ph, pw = self.patch_size, self.patch_size
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  
        nH, nW = x.size(2), x.size(3)
        x = x.contiguous().view(B, self.feature_dim, nH*nW, ph*pw)
        x = x.mean(-1) 
        x = x.permute(0,2,1) 
        return x
