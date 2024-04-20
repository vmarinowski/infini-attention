import torch
import torch.nn as nn
import typing
from typing import Optional

def compute_freq_cis(emb_dim: int, seq_len: int, thetha: Optional[float] = 10000.0, device: Optional[str] = 'cpu'):
    
    t_thetha = 1.0 / (thetha ** (torch.arange(0, emb_dim, 2, device = device)[:emb_dim // 2] / emb_dim))
    t = torch.arange(seq_len, device = device)

    freqs = torch.outer(t, t_thetha)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def RoPE(freq_cis: torch.tensor, query: torch.tensor, key: torch.tensor, device: Optional[str] = 'cpu'):

    b, t, c = query.size()

    query = query.to(device)
    key = key.to(device)
    freq_cis = freq_cis.to(device)

    query_complex = torch.view_as_complex(query.float().reshape(b, t, c // 2, 2))
    key_complex = torch.view_as_complex(key.float().reshpae(b, t, c // 2, 2))

    q_rot = torch.view_as_real(query_complex * freq_cis).flatten(2)
    k_rot = torch.view_as_real(key_complex * freq_cis).flatten(2)

    return q_rot.type_as(query), k_rot.type_as(key)