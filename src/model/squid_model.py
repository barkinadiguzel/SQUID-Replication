import torch
import torch.nn as nn
from ..backbone.encoder import Encoder
from ..memory.memory_queue import MemoryQueue
from ..modules.inpainting import InpaintingBlock
from ..modules.generators import TeacherGenerator, StudentGenerator
from ..modules.discriminator import Discriminator
from ..layers.masked_shortcut import masked_shortcut
from ..utils.anomaly import anomaly_score

class SQUIDModel(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.encoder = Encoder(embed_dim)
        self.memory = MemoryQueue(embed_dim)
        self.inpainting = InpaintingBlock(embed_dim)
        self.teacher = TeacherGenerator(embed_dim)
        self.student = StudentGenerator(embed_dim)
        self.discriminator = Discriminator()

    def forward(self, I, mu=0, sigma=1, gating_prob=0.5):
        F = self.encoder(I) 
        N = self.memory(F)   
        F_inpaint = self.inpainting(F, N)
        F_out = masked_shortcut(F, F_inpaint, gating_prob)
        I_student = self.student(F_out)
        I_teacher = self.teacher(F)
        D_student = self.discriminator(I_student)
        A = anomaly_score(self.discriminator, I_student, mu, sigma)
        return I_student, I_teacher, D_student, A  
