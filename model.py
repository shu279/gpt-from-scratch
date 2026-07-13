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

    '''
    Simple self-attention with no causal mask & MHA
    def forward(self,x):
        B, T, C = x.size()
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        score = q@k.transpose(-1,-2)
        return F.softmax(score,dim=-1)@v
    '''

    def forward(self,x):
        #x = embedding
        B, T, C = x.size()
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        #Divide embedding dimension (B,T,C) --> (B,h,T,C//h)
        head_size = C // self.h
        q = q.view(B, T, self.h, head_size).transpose(1, 2)
        k = k.view(B, T, self.h, head_size).transpose(1, 2)
        v = v.view(B, T, self.h, head_size).transpose(1, 2)

        score = q@k.transpose(-1,-2) # (B, h, Tq, Tk)
        score /= (head_size ** 0.5) # Make var = 1 so big C won't get big fluctuation

        #Causal mask
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device) #(1, 1, Tq, Tk)
        mask = torch.tril(mask)
        score = score.masked_fill(~mask,float("-inf"))

        c = F.softmax(score,dim=-1)@v #Combine across key positions (B,h,Tq,C//h)

        #Connect head dimensions
        c = c.transpose(1, 2)
        c = c.contiguous().view(B, T, C)

        return self.Wo(c) #Combine across embedding

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
    def __init__(self,config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.LN1 = nn.LayerNorm(config.C)
        self.MLP = MultiLayerPerceptron(config)
        self.LN2 = nn.LayerNorm(config.C)

    def forward(self,x):
        x = self.LN1(x + self.attention(x))
        x = self.LN2(x + self.MLP(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 5000
    block_size: int = 128
    C: int = 256
    h: int = 8
    L: int = 6
    dropout: float = 0.1
