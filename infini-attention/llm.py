import torch
import torch.nn as nn
from infini_attention import InfiniAttention
from rmsnorm import RMSNorm
import typing
from typing import Optional

class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_dim: int, dropout_rate: float, bias : Optional[bool] = False, device : Optional[str] = 'cpu'):
        super().__init__()

        self.device = device
        self.w1 = nn.Linear(emb_dim, emb_dim * 4, bias = bias, device = device)
        self.w2 = nn.Linear(emb_dim * 4, emb_dim, bias = bias, device = device)
        self.w3 = nn.Linear(emb_dim, emb_dim * 4, bias = bias, device = device)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.gelu(self.w1(x)) * self.w3(x)
        x = self.w2(x)

        return self.drop(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, seq_len: int, emb_dim: int, d_head: int, n_head: int, n_segments:int,
                 dropout_rate: float, eps: Optional[float] = 1e-5, device: Optional[str] = 'cpu'):
        super().__init__()

        self.device = device
        self.attention = InfiniAttention(seq_len, emb_dim, d_head, n_head, n_segments, update = 'delta')
        self.attn_norm = RMSNorm(emb_dim, eps, device)
        self.ffn_norm = RMSNorm(emb_dim, eps, device)
        self.ffn = PositionWiseFeedForward(emb_dim, dropout_rate, device)

    def forward(self, x):

        x = x + self.attn_norm(self.attention(x))
        x = x + self.ffn_norm(self.ffn(x))

        return x
    
class Transforyevsky(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, seq_len: int,
                 n_blocks: int, d_head: int, dropout_rate: int,
                 n_head: int, n_segments: int, eps: Optional[float] = 1e-5, 
                 device: Optional[str] = 'cpu'):
        
        self.device = device
        self.embedding = nn.Embedding(vocab_size, emb_dim, device = device)
        self.blocks = nn.Sequential(*[DecoderBlock(seq_len, emb_dim, d_head, n_head, n_segments, dropout_rate, device = device) for _ in range(n_blocks)])
        self.linear = nn.Linear(emb_dim, vocab_size, bias = False, device = device)
        self.drop = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        
        x = self.drop(self.embedding(x))
        x = self.blocks(x)

        return self.linear(x)         

