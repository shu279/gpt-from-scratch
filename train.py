import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken
from model import GPT, GPTConfig

from dataclasses import asdict

tokenizer = "o200k_base"
B = 32
T = 128
max_steps = 100
lr = 0.0003
eval_iters = 20

enc = tiktoken.get_encoding(tokenizer)

config = GPTConfig(
    V = enc.n_vocab,
    block_size = 128,
    C = 256,
    h = 8,
    L = 6,
    dropout = 0.1,
    Bc = 4,
    Br = 4,
)

with open("enwik8", "r", encoding="utf-8", newline="") as file:
    text = file.read()

tokens = enc.encode(text)
n = int(len(tokens) * 0.9)
train, val = tokens[:n], tokens[n:]

train = torch.tensor(train, dtype=torch.int32)
val = torch.tensor(val, dtype=torch.int32)

# Get random B batches - split = for train or val
def get_batch(split):
    data = train if split == 'train' else val
    ind = torch.randint(len(data) - T, (B,))
    x = torch.stack([data[i : i+T] for i in ind])
    y = torch.stack([data[i+1 : i+T+1] for i in ind])
    return x.long(), y.long() #for cross entropy


# Find cross entropy for loss
def cross_entropy(logits, targets):
    y_ind = targets.unsqueeze(-1) # (B, T, 1)
    log_softmax = logits - torch.logsumexp(logits, dim=-1, keepdim=True) # (B, T, V)
    log_prob = log_softmax.gather(dim=-1, index=y_ind).squeeze(-1) # (B, T)
    loss = -log_prob.mean()
    return loss  #item() converts 0 dim tensor to python float


# Stabler metric of model performance - not for parameter update/
def estimate_loss():
    model.eval()
    with torch.no_grad():
        train_loss, val_loss = 0.0, 0.0
        for _ in range(eval_iters):
            x,y = get_batch('train')
            x,y = x.to(device),y.to(device)

            logits = model(x)
            train_loss += cross_entropy(logits, y).item()
        
            x,y = get_batch('val')
            x,y = x.to(device),y.to(device)

            logits = model(x)
            val_loss += cross_entropy(logits, y).item()
    
    model.train()
    return train_loss/eval_iters , val_loss/eval_iters


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(config).to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

# Gradient descent
for step in range(max_steps):
    x,y = get_batch('train')
    x = x.to(device)
    y = y.to(device)
    
    logits = model(x) # (B, T, V)

    optimiser.zero_grad(set_to_none=True) # Do not accumlate gardient, just use per step gradient
    loss = cross_entropy(logits, y)
    loss.backward()
    optimiser.step()

    # Check model performance
    if step % 10 == 0:
        train_loss, val_loss = estimate_loss()
        print(
            f"step {step}: "
            f"train {train_loss:.4f}, "
            f"val {val_loss:.4f}"
        )

# Save model
checkpoint = {
    "model": model.state_dict(), # For generation / resume train
    "optimizer": optimiser.state_dict(), # For resume momentum etc
    "config": asdict(config),
    "step": max_steps,
    "tokenizer": tokenizer,
}
torch.save(checkpoint, "checkpoint.pt")