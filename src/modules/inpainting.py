import torch
import torch.nn as nn
from ..layers.transformer import PatchTransformer

class InpaintingBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.transformer = PatchTransformer(embed_dim)

    def forward(self, F, N):
        F_out = []
        for i in range(F.shape[0]):
            Fi = F[i]
            Ni = N[i]  
            F_out.append(self.transformer(Fi, Ni))
        F_out = torch.stack(F_out, dim=0)
        return F_out
