import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig
import tiktoken

'''
checkpointを読む
→ configからGPTを再構築
→ parameterを読み込む
→ promptをencode
→ 次tokenを1つsample
→ 入力末尾へ追加
→ 繰り返す
→ 全tokenをdecode
'''
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("checkpoint.pt", map_location=device)

config = GPTConfig(**checkpoint["config"])

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

enc = tiktoken.get_encoding(checkpoint["tokenizer"])