import torch
import torch.nn as nn
import typing
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: Optional[float] = 1e-5, device: Optional[str] = 'cuda'):
        super().__init__()

        self.device = device
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim, device = device))
    
    def forward(self, x):

        x = x.to(self.device)
        x_sqr = x**2
        RMS = torch.rsqrt(x_sqr.mean(dim = -1, keepdim = True) + self.eps)
        new_x = x * RMS
        new_x = new_x * self.weight

        return new_x
