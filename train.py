import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import train_bpe, encode, decode
from model import GPT, GPTConfig

config = GPTConfig(
    V = 5000,
    block_size = 128,
    C = 256,
    h = 8,
    L = 6,
    dropout = 0.1,
    Bc = 4,
    Br = 4,
)

B = 32
T = 128
max_steps = 100
lr = 0.0003

with open("enwik8", "r", encoding="utf-8", newline="") as file:
    text = file.read()

n = int(len(text) * 0.9)
train_text = text[:n]
val_text = text[n:]

merges = train_bpe(train_text[:100000], config.V - 256)
train_ids = encode(train_text, merges)
val_ids = encode(val_text, merges)

train = torch.tensor(train_ids, dtype=torch.int32)
val = torch.tensor(val_ids, dtype=torch.int32)

def get_batch(split):
    data = train if split == 'train' else val
    ind = torch.randint(len(data) - T, (B,))
    x = torch.stack([data[i : i+T] for i in ind])
    y = torch.stack([data[i+1 : i+T+1] for i in ind])
    return x.long(), y.long() #for cross entropy

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(config).to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

# Gradient descent
for step in range(max_steps):
    x,y = get_batch('train')
    x = x.to(device)
    y = y.to(device)
    
    logits = model(x) # (B, T, V)

    # Find cross entropy
    y_ind = y.unsqueeze(-1) # (B, T, 1)
    log_softmax = logits - torch.logsumexp(logits, -1, keepdim=True) # (B, T, V)
    log_prob = log_softmax.gather(dim=-1, index=y_ind).squeeze(-1) # (B, T)
    loss = -log_prob.mean()

    optimiser.zero_grad(set_to_none=True) # Do not accumlate gardient, just use per step gradient
    loss.backward()
    optimiser.step()

    if step % 10 == 0:
        print(step, loss.item())

# Stabler metric of model performance - not for parameter update/
def estimate_loss():
    model.eval()
    with torch.no_grad():
        x,y = get_batch('train')
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        train_loss = F.cross_entropy(logits.reshape(-1,config.V), y.reshape(-1))
    
        x,y = get_batch('valid')
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        valid_loss = F.cross_entropy(logits.reshape(-1,config.V), y.reshape(-1))
    
    return train_loss, valid_loss





    


    



