import torch

def masked_shortcut(F, F_inpaint, gating_prob=0.5):
    if F.device != F_inpaint.device:
        F_inpaint = F_inpaint.to(F.device)
    delta = torch.bernoulli(torch.full(F.shape[:1], gating_prob, device=F.device))
    delta = delta.view(-1, *([1]*(F.dim()-1)))
    F_out = (1 - delta) * F + delta * F_inpaint
    return F_out
