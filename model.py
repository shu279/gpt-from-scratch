import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        C = config.C
        assert config.C % config.h == 0
        self.h = config.h
        self.dropout = config.dropout

        self.Wq = nn.Linear(C,C)
        self.Wk = nn.Linear(C,C)
        self.Wv = nn.Linear(C,C)
        self.Wo = nn.Linear(C,C)

    def forward(self,x):
        # x = embedding
        B, T, C = x.size()
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        # Divide embedding dimension (B,T,C) --> (B,h,T,C//h)
        head_size = C // self.h
        q = q.view(B, T, self.h, head_size).transpose(1, 2)
        k = k.view(B, T, self.h, head_size).transpose(1, 2)
        v = v.view(B, T, self.h, head_size).transpose(1, 2)

        q,k = RoPE(q,k) # Rotate q/k

        dropout_p=self.dropout if self.training else 0.0

        c = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p) #(B,h,T,head_size)

        # Connect head dimensions
        c = c.transpose(1, 2).contiguous()
        c = c.view(B, T, C)

        return self.Wo(c) # Combine across embedding


# Rotate q/k according to token position -> learn relationship by relative position
def RoPE(q,k):
    B, h, T, head_size = q.shape
    assert head_size % 2 == 0

    # Pair-up each embedding (q0, q1), (q2, q3),...
    q_even, q_odd = q[...,0::2], q[...,1::2] # (B, h, T, head_size/2)
    k_even, k_odd = k[...,0::2], k[...,1::2]

    # Frequency = rotation angle per 1 token
    # Change frequency between embedding pairs
    # pair 0 -> 1.00 rad per position --> sensitive for fine/local s-t change
    # pair 1 -> 0.10 rad per position
    # pair 2 -> 0.01 rad per position --> sensitive for coarse s-t change

    pos = torch.arange(0, T, device = q.device)
    pair_idx = torch.arange(0, head_size//2, device = q.device)
    freq = 1 / (10000 ** (2 * pair_idx / head_size))

    angles = pos[:, None] * freq[None, :] # Broadcast and element-wise multiply --> (T, head_size/2)
    angles = angles[None, None, :, :]

    # Rotate q/k along token position
    q_even_new = q_even * torch.cos(angles) - q_odd * torch.sin(angles)
    q_odd_new = q_even * torch.sin(angles) + q_odd * torch.cos(angles)
    q_rotated = torch.stack((q_even_new, q_odd_new), dim=-1).flatten(-2)

    k_even_new = k_even * torch.cos(angles) - k_odd * torch.sin(angles)
    k_odd_new = k_even * torch.sin(angles) + k_odd * torch.cos(angles)
    k_rotated = torch.stack((k_even_new, k_odd_new), dim=-1).flatten(-2)

    return q_rotated, k_rotated


class MultiLayerPerceptron(nn.Module):
    def __init__(self,config):
        super().__init__()
        C = config.C
        self.W1 = nn.Linear(C,4*C)
        self.W2 = nn.Linear(4*C,C)

    def forward(self, x):
        x = self.W1(x)
        x = F.gelu(x)
        x = self.W2(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.LN1 = nn.LayerNorm(config.C)
        self.MLP = MultiLayerPerceptron(config)
        self.LN2 = nn.LayerNorm(config.C)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = x + self.dropout(self.attention(self.LN1(x))) # reduce overfit on attention feature + avoid breaking residual connection
        x = x + self.dropout(self.MLP(self.LN2(x)))
        return x


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.block_size = config.block_size
        self.Wt = nn.Embedding(config.V, config.C)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.L)])
        self.LN = nn.LayerNorm(config.C) #Remove fluctuation by repeated residual connection
        self.Wo = nn.Linear(config.C, config.V)

        self.apply(self._init_weights) # default linear/embedding weight is too big  
        self.Wo.weight = self.Wt.weight # Weight tying by sharing weight 

    def forward(self,x):
        B, T = x.shape
        assert T <= self.block_size

        x = self.Wt(x) # (B, T, C)
        for block in self.blocks:
            x = block(x) # (B, T, C)
        x = self.LN(x) 
        return self.Wo(x) # (B, T, V)
    
    # default linear/embedding weight is too big  
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

@dataclass
class GPTConfig:
    V: int
    block_size: int
    C: int
    h: int
    L: int
    dropout: float

    #Flash attention
    Bc: int
    Br: int
