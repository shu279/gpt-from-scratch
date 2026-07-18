import torch
from tokenizer import train_bpe, encode, decode
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=5000,
    block_size=128,
    C=256,
    h=8,
    L=6,
    dropout=0.1,
    Bc=4,
    Br=4,
)

with open("enwik8", "r", encoding="utf-8", newline="") as file:
    text = file.read()

n = int(len(text) * 0.9)
train_text = text[:n]
val_text = text[n:]

merges = train_bpe(train_text, config.vocab_size - 256)
train_ids = encode(train_text, merges)
val_ids = encode(val_text, merges)

train = torch.tensor(train_ids)
val = torch.tensor(val_ids)

model = GPT(config)

def get_batch():
    