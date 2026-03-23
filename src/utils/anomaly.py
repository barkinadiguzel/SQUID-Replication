import torch
import torch.nn.functional as F

def anomaly_score(D, reconstructed, mu, sigma):
    score = D(reconstructed)
    A = torch.sigmoid((score - mu) / sigma)
    return A
