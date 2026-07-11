import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config.C
        self.Wq = nn.Linear(C,C)
        self.Wk = nn.Linear(C,C)
        self.Wv = nn.Linear(C,C)

    def forward(self,x):
        #x = embedding
        B, T, C = x.size()
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        score = F.softmax(q@k,dim=-1)
        c = score@v

@dataclass
class GPTConfig:
    vocab_size: int = 5000
    block_size: int = 128
    C: int = 256
    nh: int = 8
    L: int = 6
    dropout: float = 0.1