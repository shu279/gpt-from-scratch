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

        q,k = RoPE(q,k) # Rotate q/k

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
        self.Wo = nn.Linear(C, C)

    def forward(self, x):
        B, T, C = x.size()
        h = self.h
        d = C//self.h # Head size

        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        # Multi-head attention
        q = q.view(B, T, h, d).transpose(1,2)
        k = k.view(B, T, h, d).transpose(1,2)
        v = v.view(B, T, h, d).transpose(1,2)

        q,k = RoPE(q,k) # Rotate q/k

        output = q.new_zeros(B, h, T, d)

        Br = self.Br
        for i in range(0, T, Br):
            q_block = q[:, :, i:i+Br, :] #(B,h,Br,d)
            Br = q_block.shape[-2]

            m = q_block.new_full((B, h, Br, 1), float("-inf"))
            l = q_block.new_zeros((B, h, Br, 1))
            o = q_block.new_zeros((B, h, Br, d))

            Bc = self.Bc
            for j in range(0, T, Bc):
                k_block = k[:, :, j:j+Bc, :] #(B,h,Bc,d)
                v_block = v[:, :, j:j+Bc, :] #(B,h,Bc,d)
                Bc = k_block.shape[-2]

                score = q_block @ k_block.transpose(-1,-2) #(B,h,Br,Bc)
                score /= d ** 0.5

                # Causal mask
                q_pos = torch.arange(Br, device=q.device) + i
                k_pos = torch.arange(Bc, device=q.device) + j
                mask = k_pos[None, :] > q_pos[:, None]
                score = score.masked_fill(mask, float("-inf"))

                m_block = torch.amax(score, dim=-1, keepdim=True) # Find maximum score in block
                m_new = torch.maximum(m, m_block) # Maximum score globally

                correction = torch.exp(m - m_new) # Rescale for new gloabl maximum score
                p = torch.exp(score - m_new) # Scores in current block to add
                l = l * correction + p.sum(dim=-1, keepdim=True)
                o = o * correction + p @ v_block
                m = m_new

            output[:, :, i:i+Br, :] = o / l

        # Connect heads back
        output = output.transpose(1,2).contiguous()
        output = output.view(B,T,C)
        output = self.Wo(output)

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
        self.Wt = nn.Embedding(config.V, config.C)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.L)])
        self.LN = nn.LayerNorm(config.C) #Remove fluctuation by repeated residual connection
        self.Wo = nn.Linear(config.C, config.V)
        self.block_size = config.block_size

    def forward(self,x):
        # input sequence (B, T)
        x = self.Wt(x) # (B, T, C)
        for block in self.blocks:
            x = block(x) # (B, T, C)
        x = self.LN(x) 
        return self.Wo(x) # (B, T, V)


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
