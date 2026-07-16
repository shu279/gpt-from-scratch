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

        self.Wq = nn.Linear(C,C)
        self.Wk = nn.Linear(C,C)
        self.Wv = nn.Linear(C,C)
        self.Wo = nn.Linear(C,C)

    # Simple self-attention
    def plain_forward(self,x):
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        score = q@k.transpose(-1,-2)
        return F.softmax(score,dim=-1)@v

    # Added causal mask, MHA, RoPE
    def forward(self,x):
        # x = embedding
        B, T, C = x.size()
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        # Divide embedding dimension (B,T,C) --> (B,h,T,C//h)
        head_size = C // self.h
        q = q.view(B, T, self.h, head_size).transpose(1, 2)
        k = k.view(B, T, self.h, head_size).transpose(1, 2)
        v = v.view(B, T, self.h, head_size).transpose(1, 2)

        q,k = self.RoPE(q,k) # Rotate q/k

        score = q@k.transpose(-1,-2) # (B, h, Tq, Tk)
        score /= (head_size ** 0.5) # Make var = 1 so big C won't get big fluctuation

        # Causal mask
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device) #(1, 1, Tq, Tk)
        mask = torch.tril(mask)
        score = score.masked_fill(~mask,float("-inf"))

        c = F.softmax(score,dim=-1)@v # Combine across key positions (B,h,Tq,C//h)

        # Connect head dimensions
        c = c.transpose(1, 2)
        c = c.contiguous().view(B, T, C)

        return self.Wo(c) # Combine across embedding


    # Rotate q/k according to token position -> learn relationship by relative position
    def RoPE(self,q,k):
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

class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        C = config.C
        assert config.C % config.h == 0
        self.h = config.h
        self.Bc = config.Bc
        self.Br = config.Br

        self.Wq = nn.Linear(C, C)
        self.Wk = nn.Linear(C, C)
        self.Wv = nn.Linear(C, C)

    def forward(self, x):
        B, T, C = x.size()
        h = self.h
        d = C//self.h # Head size
        Br = self.Br
        Bc = self.Bc

        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        q = q.view(B, T, h, d).transpose(1,2)
        k = k.view(B, T, h, d).transpose(1,2)
        v = v.view(B, T, h, d).transpose(1,2)
        output = q.new_zeros(B, h, T, d)

        for i in range(0, T, Br):
            q_block = q[:, :, i:i+Br, :] #(B,h,Br,d)
            R = q_block.shape[-2]

            m = q_block.new_full((B, h, R, 1), float("-inf"))
            l = q_block.new_zeros((B, h, R, 1))
            o = q_block.new_zeros((B, h, R, d))

            for j in range(0, T, Bc):
                k_block = k[:, :, j:j+Bc, :] #(B,h,Bc,d)
                v_block = v[:, :, j:j+Bc, :] #(B,h,Bc,d)
                score = q_block @ k_block.transpose(-1,-2) #(B,h,Br,Bc)
                score /= d ** 0.5

                max_score = torch.amax(score, dim=-1, keepdim=True)
                m_new = torch.maximum(m,max_score)

                correction = torch.exp(m - m_new)
                p = torch.exp(score - m_new)

                l = l * correction + p.sum(dim=-1, keepdim=True)
                o = o * correction + p @ v_block
                m = m_new

            output[:, :, i:i+Br, :] = o / l

        return output


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.LN1 = nn.LayerNorm(config.C)
        self.MLP = MultiLayerPerceptron(config)
        self.LN2 = nn.LayerNorm(config.C)

    def forward(self,x):
        x = x + self.attention(self.LN1(x))
        x = x + self.MLP(self.LN2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.Wt = nn.Embedding(config.vocab_size, config.C)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.L)])
        self.LN = nn.LayerNorm(config.C) #Remove fluctuation by repeated residual connection
        self.Wo = nn.Linear(config.C, config.vocab_size)

    def forward(self,x):
        # x = input sequence (B, T)
        x = self.Wt(x)
        for block in self.blocks:
            x = block(x)
        x = self.LN(x)
        return self.Wo(x)

@dataclass
class GPTConfig:
    vocab_size: int = 5000
    block_size: int = 128
    C: int = 256
    h: int = 8
    L: int = 6
    dropout: float = 0.1

    #Flash attention
    Bc: int = 4
    Br: int = 4
