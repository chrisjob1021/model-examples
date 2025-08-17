# Convolutional Social Pooling for Vehicle Trajectory Prediction

Implementation of the paper "Convolutional Social Pooling for Vehicle Trajectory Prediction" (Deo & Trivedi, 2018) from https://arxiv.org/abs/1805.06771

## Overview

This project implements an LSTM encoder-decoder model with convolutional social pooling for predicting vehicle trajectories in highway traffic scenarios. The model uses the motion patterns of surrounding vehicles to improve trajectory prediction accuracy.

## Architecture

- **LSTM Encoder**: Processes historical trajectory sequences
- **Convolutional Social Pooling**: Captures spatial interactions between vehicles using CNN layers
- **LSTM Decoder**: Generates multi-modal trajectory predictions
- **Multi-Modal Output**: Predicts 6 different possible future trajectories with associated probabilities

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The model is designed to work with the NGSIM (Next Generation Simulation) dataset:
- US-101 Highway Dataset
- I-80 Highway Dataset

Download the dataset from: https://www.fhwa.dot.gov/publications/research/operations/07030/

## Training

```bash
python train.py \
    --data_path path/to/ngsim_data.txt \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --encoder_dim 128 \
    --num_modes 6
```

### Training Arguments

- `--data_path`: Path to NGSIM dataset file (required)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--encoder_dim`: Hidden dimension for encoder (default: 128)
- `--num_modes`: Number of prediction modes (default: 6)
- `--hist_len`: Length of history trajectory in frames (default: 30)
- `--pred_len`: Length of predicted trajectory in frames (default: 50)
- `--save_dir`: Directory to save model checkpoints (default: selfdriving/checkpoints)
- `--log_dir`: Directory for tensorboard logs (default: selfdriving/logs)

## Evaluation

```bash
python evaluate.py \
    --checkpoint selfdriving/checkpoints/best_model.pth \
    --data_path path/to/ngsim_data.txt \
    --visualize \
    --num_samples 10
```

### Evaluation Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--data_path`: Path to NGSIM dataset file (required)
- `--visualize`: Generate visualization plots
- `--num_samples`: Number of samples to visualize (default: 10)

## Model Components

### 1. LSTM Encoder-Decoder (`models/encoder_decoder.py`)
- Encodes vehicle trajectory history
- Decodes future trajectories with multiple modes

### 2. Convolutional Social Pooling (`models/social_pooling.py`)
- Creates spatial grid representation of neighboring vehicles
- Applies convolutional layers to extract social features
- Uses max pooling to aggregate spatial information

### 3. Trajectory Model (`models/trajectory_model.py`)
- Combines encoder-decoder with social pooling
- Implements multi-modal loss function
- Handles neighbor data processing

### 4. NGSIM Dataset (`data/ngsim_dataset.py`)
- Preprocesses NGSIM trajectory data
- Handles neighbor vehicle extraction
- Creates train/validation splits

## Results

The model outputs:
- **Min ADE**: Minimum Average Displacement Error across all modes
- **Min FDE**: Minimum Final Displacement Error across all modes
- **Mode Probabilities**: Likelihood of each predicted trajectory mode

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir selfdriving/logs
```

## Paper Reference

```bibtex
@inproceedings{deo2018convolutional,
  title={Convolutional social pooling for vehicle trajectory prediction},
  author={Deo, Nachiket and Trivedi, Mohan M},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={1468--1476},
  year={2018}
}
```