# GPT from Scratch

A small educational implementation of a decoder-only Transformer in PyTorch.

## Implementations

### Practical

- `model.py`: GPT with rotary position embeddings (RoPE) and KV-cached attention
- `train.py`: tiktoken-based training loop
- `generate.py`: interactive autoregressive generation from a checkpoint

### Simple

- `simple_model.py`: handwritten causal multi-head attention and a reference tiled FlashAttention implementation
- `simple_train.py`: minimal training loop with handwritten cross-entropy
- `simple_tokenizer.py`: byte-level BPE tokenizer

## Architecture

```text
token embedding
    ↓
[ LayerNorm → causal attention + RoPE → residual
  LayerNorm → GELU MLP              → residual ] × L
    ↓
final LayerNorm → vocabulary logits
```

## Setup

```bash
pip install torch tiktoken
```

The scripts are intentionally configured directly in source rather than through a command-line interface.

- `train.py` reads an `enwik8` file from the repository root and writes `checkpoint.pt`.
- `generate.py` loads `checkpoint.pt` and asks for a prompt interactively.
- `simple_train.py` also reads `enwik8` and writes `checkpoint_simple.pt`.

The `FlashAttention` class in `simple_model.py` is included as a readable reference implementation; the default simple GPT block uses regular causal self-attention.
