# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational PyTorch implementations of deep learning architectures: GPT-2 transformer for language modeling, CNN with PReLU for ImageNet classification, LSTM pointer networks, RNNsearch attention, and trajectory prediction for self-driving.

## Commands

```bash
# Setup
./scripts/setup_venv.sh              # Create .venv and install dependencies

# CNN Training (from cnn/ directory)
python train_cnn_imagenet.py         # Train on ImageNet-1k
python evaluate_model.py             # Evaluate top-1/top-5 accuracy
python visualize_activations.py      # Visualize learned features
python visualize_gradients.py        # Monitor gradient flow
python upload_to_huggingface.py      # Upload model to HuggingFace Hub
  --repo-name MODEL_NAME             #   Example: --repo-name cnn-prelu-imagenet
  --checkpoint PATH                  #   Optional: specify checkpoint path
  --top1-acc ACC --top5-acc ACC      #   Optional: include accuracy in model card

# GPT-2 Training (from transformer/ directory)
python train_gpt2.py                 # Train Stage 1: Pretraining on Dolma
python train_gpt2.py --stage midtrain # Train Stage 2: Mid-training annealing
python train_gpt2.py --stage sft     # Train Stage 3: SFT on Tulu-3
python train_gpt2.py --stage all     # Run all 3 stages sequentially
python train_gpt2.py --validate-weights # Validate impl by loading HF GPT-2 weights
python evaluate_model.py             # Evaluate perplexity on WikiText-2
python evaluate_model.py --generate  # Evaluate + generate text samples
python upload_to_huggingface.py      # Upload model to HuggingFace Hub
  --repo-name MODEL_NAME             #   Example: --repo-name gpt2-124m-dolma
  --stage STAGE                      #   Training stage: pretrain/midtrain/sft
  --perplexity PPL                   #   Optional: include perplexity in model card

# Monitoring
./scripts/start_tensorboard.sh       # TensorBoard on localhost:6006

# Jupyter notebooks
jupyter notebook                     # Start Jupyter server
```

## Architecture

### CNN Pipeline (`cnn/`)
- `prelu_cnn.py` - Model architecture: `ManualConv2d`, `ConvAct`, `ResidualBlock`, `CNN`, `CNNTrainer`
- `train_cnn_imagenet.py` - Training with MixUp/CutMix via `MixupCutmixCollator`
- Manual conv/pooling implementations alongside PyTorch builtins (toggle with `use_builtin_conv`)

### Transformer Pipeline (`transformer/`)
- `gpt2.py` - Model architecture: `GPT2Config`, `ManualCausalSelfAttention`, `GPT2MLP`, `GPT2Block`, `GPT2`, `GPT2Trainer`
- `train_gpt2.py` - 3-stage training: pretraining (Dolma), mid-training (annealing), SFT (Tulu-3)
- `evaluate_model.py` - Perplexity evaluation and text generation
- `upload_to_huggingface.py` - HuggingFace Hub upload with model card
- Manual attention implementation alongside PyTorch `scaled_dot_product_attention` (toggle with `use_builtin_attn`)
- `from_huggingface()` loads `openai-community/gpt2` weights for implementation validation

### Shared Utilities (`shared_utils/`)
- `trainer.py` - `ModelTrainer` wrapper around HuggingFace `Trainer`
- `dataset_processor.py` - Chunked dataset preprocessing for large datasets
- `checkpoint_utils.py` - Model checkpoint management

### Key Patterns
- HuggingFace Transformers `Trainer` subclassing for custom loss (soft labels from MixUp/CutMix, causal LM)
- Manual implementations for education with `use_builtin=True` fallback for production
- Channel-wise vs channel-shared PReLU activation (configurable via `prelu_channel_wise`)
- Weight tying between token embeddings and LM head (GPT-2)
- Conv1D↔Linear weight transpose for HuggingFace GPT-2 weight compatibility

## Data

- ImageNet-1k loaded via HuggingFace datasets (CNN)
- Dolma, dolma3_dolmino_mix, tulu-3-sft-mixture loaded via HuggingFace datasets (Transformer)
- Transforms applied on-the-fly via torchvision (preprocessing optional)
- Training logs saved to `cnn/logs/<date>_<timestamp>/` and `transformer/results/gpt2_results/`
- Checkpoints saved to `cnn/results/` and `transformer/results/`
