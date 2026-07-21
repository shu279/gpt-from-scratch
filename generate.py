import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"
temperature = 0.8
generation_length = 100

# Load trained model
checkpoint = torch.load("checkpoint.pt", map_location=device)

config = GPTConfig(**checkpoint["config"])

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

prompt = input("Prompt: ")
if not prompt: raise ValueError("Prompt must not be empty")

enc = tiktoken.get_encoding(checkpoint["tokenizer"])

tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0) # (1,T)

# Autoregressive loop
with torch.no_grad():
    for _ in range(generation_length):
        tokens = tokens[:, -config.block_size:] # remove old context - only use recent block_size part
        logits = model(tokens)
        next_logits = logits[:, -1, :]
        prob = torch.softmax(next_logits, dim = -1) # use logit of last token to find prob for each vocab
        next_token = torch.multinomial(prob, num_samples=1) # randomly pick vocab by prob
        tokens = torch.cat((tokens, next_token), dim=1) # add predicted token
    
print(enc.decode(tokens[:, -generation_length:]))





