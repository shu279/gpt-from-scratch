import torch
import torch.nn.functional as F

import tiktoken
from model import GPT, GPTConfig

from dataclasses import asdict

tokenizer = "gpt2"
enc = tiktoken.get_encoding(tokenizer)

config = GPTConfig(
    V = enc.n_vocab,
    block_size = 512,
    C = 512,
    h = 8,
    L = 8,
    dropout = 0.1,
    Bc = 4,
    Br = 4,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
B = 24
T = config.block_size
lr = 0.0003
eval_iters = 10
eval_interval = 200

with open("enwik8", "r", encoding="utf-8", newline="") as file:
    text = file.read()

tokens = enc.encode(text)
n = int(len(tokens) * 0.9)
train, val = tokens[:n], tokens[n:]

train = torch.tensor(train, dtype=torch.int32)
val = torch.tensor(val, dtype=torch.int32)

max_steps = (3*n) // (B*T) # Able to see every data roughly 3 times - (3 * 26112214) / 12288 ~= 6400


model = GPT(config).to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
best_val_loss = float("inf")

checkpoint = {
    "model": model.state_dict(), # For generation / resume train
    "optimizer": optimiser.state_dict(), # For resume momentum etc
    "config": asdict(config),
    "step": max_steps,
    "tokenizer": tokenizer,
}


# Get random B batches - split = for train or val
def get_batch(split):
    data = train if split == 'train' else val
    ind = torch.randint(len(data) - T, (B,))
    x = torch.stack([data[i : i+T] for i in ind])
    y = torch.stack([data[i+1 : i+T+1] for i in ind])
    return x.long(), y.long() #for cross entropy


# Stabler metric of model performance - not for parameter update
def estimate_loss():
    model.eval()
    with torch.no_grad():
        train_loss, val_loss = 0.0, 0.0
        for _ in range(eval_iters):
            x,y = get_batch('train')
            x,y = x.to(device),y.to(device)

            logits = model(x)
            train_loss += F.cross_entropy(logits.view(-1, config.V), y.view(-1)).item()
        
            x,y = get_batch('val')
            x,y = x.to(device),y.to(device)

            logits = model(x)
            val_loss +=  F.cross_entropy(logits.view(-1, config.V), y.view(-1)).item()

            # save checkpoint when val loss improve
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, "checkpoint.pt")
    
    model.train()
    return train_loss/eval_iters , val_loss/eval_iters


# Gradient descent
for step in range(max_steps):
    x,y = get_batch('train')
    x = x.to(device)
    y = y.to(device)
    
    logits = model(x) # (B, T, V)

    optimiser.zero_grad(set_to_none=True) # Do not accumlate gardient, just use per step gradient
    loss = F.cross_entropy(logits.view(-1, config.V), y.view(-1))
    loss.backward()
    optimiser.step()

    # Check model performance
    if step % eval_interval == 0:
        train_loss, val_loss = estimate_loss()
        print(
            f"step {step}: "
            f"train {train_loss:.4f}, "
            f"val {val_loss:.4f}"
        )

torch.save(checkpoint, "checkpoint.pt")

print("Training is finished!")