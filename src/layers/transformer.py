import torch
import torch.nn as nn

class PatchTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.attn(query.unsqueeze(1), key_value.unsqueeze(1), key_value.unsqueeze(1))
        x = self.norm1(query + attn_output.squeeze(1))
        x = self.norm2(x + self.ffn(x))
        return x
