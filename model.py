import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

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

    def forward(self,x, past_kv=None, use_cache=False):
        # x = embedding
        B, T, C = x.size()
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        # Divide embedding dimension (B,T,C) --> (B,h,T,C//h)
        head_size = C // self.h
        q = q.view(B, T, self.h, head_size).transpose(1, 2)
        k = k.view(B, T, self.h, head_size).transpose(1, 2)
        v = v.view(B, T, self.h, head_size).transpose(1, 2)

        # Check past T - how much kv are cached
        past_length = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_length = past_k.size(2)
            assert T == 1 # When using past_kv, T must be 1 in inference

        q,k = RoPE(q,k, offset=past_length) # Rotate q/k

        # Add past key/value for inference
        if past_kv is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        dropout_p=self.dropout if self.training else 0.0

        c = F.scaled_dot_product_attention(q, k, v, is_causal=past_kv is None, dropout_p=dropout_p) #(B,h,T,head_size)

        # Connect head dimensions
        c = c.transpose(1, 2).contiguous()
        c = c.view(B, T, C)

        output = self.Wo(c) # Combine across embedding
        if use_cache:
            return output, (k, v)

        return output


# Rotate q/k according to token position -> learn relationship by relative position
def RoPE(q,k, offset=0):
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

    pos = torch.arange(offset, offset + T, device = q.device)
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

    # Adapted for kv caching
    def forward(self,x, past_kv=None, use_cache=False):
        attn_res = self.attention(self.LN1(x), past_kv=past_kv, use_cache=use_cache)

        if use_cache:
            attn_res, kv = attn_res
        else:
            kv = None

        x = x + self.dropout(attn_res) # reduce overfit on attention feature + avoid breaking residual connection
        x = x + self.dropout(self.MLP(self.LN2(x)))

        if use_cache:
            return x, kv

        return x


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.block_size = config.block_size
        self.Wt = nn.Embedding(config.V, config.C)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.L)])
        self.LN = nn.LayerNorm(config.C) #Remove fluctuation by repeated residual connection
        self.Wo = nn.Linear(config.C, config.V)

        self.L = config.L
        self.apply(self._init_weights) # default linear/embedding weight is too big
        self.Wo.weight = self.Wt.weight # Weight tying by sharing weight

    # Adapted for kv caching
    def forward(self, x, past_kvs=None, use_cache=False):
        B, T = x.shape
        assert T <= self.block_size


        if past_kvs is None:
            past_kvs = [None] * self.L
        else:
            assert len(past_kvs) == self.L
            past_length = past_kvs[0][0].size(2)
            assert T + past_length <= self.block_size

        x = self.Wt(x) # (B, T, C)
        kv = []

        for block, past_kv in zip(self.blocks, past_kvs):
            x = block(x, past_kv=past_kv, use_cache=use_cache) # (B, T, C)
            if use_cache:
                x, new_kv = x
                kv.append(new_kv)
            else:
                kv.append(None)

        x = self.LN(x) 
        logits = self.Wo(x) # (B, T, V)

        if use_cache:
            return logits, kv

        return logits

    
    # default linear/embedding weight is too big  
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02 / math.sqrt(2 * self.L)) #prevent residual stream variance grow
            if module.bias is not None: nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02 / math.sqrt(2 * self.L))

@dataclass
class GPTConfig:
    V: int
    block_size: int
    C: int
    h: int
    L: int
    dropout: float
