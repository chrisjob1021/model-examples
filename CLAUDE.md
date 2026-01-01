# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational PyTorch implementations of deep learning architectures: CNN with PReLU for ImageNet classification, LSTM pointer networks, RNNsearch attention, and trajectory prediction for self-driving.

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

### Shared Utilities (`shared_utils/`)
- `trainer.py` - `ModelTrainer` wrapper around HuggingFace `Trainer`
- `dataset_processor.py` - Chunked dataset preprocessing for large datasets
- `checkpoint_utils.py` - Model checkpoint management

### Key Patterns
- HuggingFace Transformers `Trainer` subclassing for custom loss (soft labels from MixUp/CutMix)
- Manual implementations for education with `use_builtin=True` fallback for production
- Channel-wise vs channel-shared PReLU activation (configurable via `prelu_channel_wise`)

## Data

- ImageNet-1k loaded via HuggingFace datasets
- Transforms applied on-the-fly via torchvision (preprocessing optional)
- Training logs saved to `cnn/logs/<date>_<timestamp>/`
- Checkpoints saved to `cnn/results/`
