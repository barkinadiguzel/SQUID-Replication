import torch
import torch.nn.functional as F

def gumbel_shrinkage(similarity, top_k=5, temperature=1.0):
    B, M = similarity.shape
    topk_val, topk_idx = similarity.topk(top_k, dim=-1)
    mask = torch.zeros_like(similarity)
    mask.scatter_(1, topk_idx, 1.0)
    w = F.softmax(similarity * mask / temperature, dim=-1)
    return w
