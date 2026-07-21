# GPT from Scratch

A small GPT implementation built from scratch.

Practical implementation
- `model.py`: GPT with RoPE and PyTorch SDPA
- `train.py`: training pipeline using tiktoken
- `generate.py`: autoregressive text generation from a checkpoint

Simple implementation
- `simple_model.py`: GPT with handwritten causal attention and FlashAttention
- `simple_tokenizer.py`: byte-level BPE tokenizer
- `simple_train.py`: training loop
