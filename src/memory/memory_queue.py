import torch
import torch.nn as nn

class MemoryQueue(nn.Module):
    def __init__(self, num_locations, memory_size=128, feature_dim=256):
        super().__init__()
        self.num_locations = num_locations
        self.memory_size = memory_size
        self.feature_dim = feature_dim

        self.register_buffer('queue', torch.zeros(num_locations, memory_size, feature_dim))
        self.register_buffer('ptr', torch.zeros(num_locations, dtype=torch.long))

    def forward(self, patch_features):
        B, num_patches, F = patch_features.shape
        N = torch.zeros_like(patch_features)

        for i in range(num_patches):
            mem_feats = self.queue[i]  
            sim = torch.matmul(patch_features[:,i,:], mem_feats.T)  
            topk = sim.topk(5, dim=-1)[1]  
            N[:,i,:] = mem_feats[topk[:,0]]  
        return N

    def update_queue(self, patch_features):
        B, num_patches, F = patch_features.shape
        for i in range(num_patches):
            for b in range(B):
                ptr = self.ptr[i].item()
                self.queue[i, ptr] = patch_features[b,i]
                self.ptr[i] = (ptr + 1) % self.memory_size
