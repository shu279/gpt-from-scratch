# GPT from Scratch

A small educational project for implementing a GPT-style language model with
PyTorch.

## Current status

The repository currently contains:

- `model.py`: an initial causal self-attention module and GPT configuration.
- `train.py`: a placeholder for the training pipeline.

The model and training loop are still under development and are not yet ready
for end-to-end training.

## Requirements

- Python 3.10 or newer
- PyTorch

Create a virtual environment and install PyTorch before running the code. See
the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
for the command appropriate for your platform.

## Development

Check that the Python files compile:

```bash
python3 -m py_compile model.py train.py
```

## Roadmap

- Complete causal multi-head self-attention.
- Add transformer blocks and the GPT model.
- Implement data loading and tokenization.
- Add a training and evaluation loop.
- Add tests and reproducible configuration.
