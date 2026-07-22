import torch

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
tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1,T)
prompt_length = tokens.size(1)

# Autoregressive loop
with torch.no_grad():
    for _ in range(generation_length):
        recent_tokens = tokens[:, -config.block_size:] # remove old context
        logits = model(recent_tokens)

        next_logits = logits[:, -1, :] / temperature # temp < 1 <=> sharper distribution
        prob = torch.softmax(next_logits, dim = -1) # use logit of last token to find prob for each vocab

        next_token = torch.multinomial(prob, num_samples=1) # randomly pick vocab by prob
        tokens = torch.cat((tokens, next_token), dim=1) # add predicted token

# Print generated text
output_tokens = tokens[0, prompt_length:].tolist()
print(enc.decode(output_tokens))





