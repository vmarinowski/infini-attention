# Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention
## Overview
`Infini-attention` ([https://arxiv.org/abs/2404.07143](https://arxiv.org/abs/2404.07143)) is an efficient method presented by Google for an alternative `MultiHeadedAttention`.
The Infini-attention incorporates
a compressive memory into the vanilla attention mechanism and builds
in both masked local attention and long-term linear attention mechanisms
in a single Transformer block. This is an unofficial PyTorch implementation of the paper.
## How to use
- Clone the repository
```bash
 git clone https://github.com/vmarinowski/infini-attention.git
```
- Import necessary libraries
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import RoPE
from infini_attention.infini_attention import InfiniAttenion
```
```python
"""
Args:
seq_len - Sequence length of the input,
emb_dim - Embedding dimension of the input,
d_head - dimension of the each head,
n_head - number of heads,
n_segments - Number of segments (must be, seq_len % n_segments == 0),
use_rope - Whether use RoPE or not, by default it's set to True,
is_causal - Use causal masking or not, by default it's set to True,
update - 'delta' or 'linear', by default it's set to 'delta', 
device - 'cuda' or 'cpu', by default it's set to cpu 
"""

seq_len = 256
emb_dim = 128
n_head = 4
d_head = emb_dim // n_head
n_segments = 4 #seq_len must be divisible to n_segments 
batch_size = 1

x = torch.randn(batch_size, seq_len, emb_dim)

Attention = InfiniAttention(seq_len, emb_dim, d_head, n_head, n_segments)
output = Attention(x)
```
- If you want to change device type:
```python
Attention = InfiniAttention(seq_len, emb_dim, d_head, n_head, n_segments, device = 'cuda') #By default it's set to cpu.
```
- By default causal masking is True and memory update is linear:
```python
Attention = InfiniAttention(seq_len, emb_dim, d_head, n_head, n_segments, is_causal = False, update = 'delta')
```
