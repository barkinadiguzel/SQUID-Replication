import torch
import torch.nn as nn
import torch.nn.functional as F

class SQUIDLoss(nn.Module):
    def __init__(self, lambda_t=1.0, lambda_s=1.0, lambda_dist=1.0, lambda_gen=1.0, lambda_dis=1.0):
        super().__init__()
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.lambda_dist = lambda_dist
        self.lambda_gen = lambda_gen
        self.lambda_dis = lambda_dis
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()  

    def forward(self, I, Gt_I, Gs_I, Ft, Fs, D_real, D_fake):
        # Teacher reconstruction loss
        Lt = self.mse(Gt_I, I)

        # Student reconstruction loss
        Ls = self.mse(Gs_I, I)

        # Distillation loss 
        Ldist = 0
        for ft, fs in zip(Ft, Fs):
            Ldist += self.mse(fs, ft)

        # Generator adversarial loss (student)
        Lgen = F.binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake))

        # Discriminator loss
        Ldis = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real)) + \
               F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))

        # Weighted sum
        total_loss = (self.lambda_t * Lt +
                      self.lambda_s * Ls +
                      self.lambda_dist * Ldist +
                      self.lambda_gen * Lgen -
                      self.lambda_dis * Ldis)  

        return {
            "Lt": Lt,
            "Ls": Ls,
            "Ldist": Ldist,
            "Lgen": Lgen,
            "Ldis": Ldis,
            "total_loss": total_loss
        }
