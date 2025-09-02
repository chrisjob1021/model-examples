# Convolutional Social Pooling for Vehicle Trajectory Prediction

Implementation of the paper "Convolutional Social Pooling for Vehicle Trajectory Prediction" (Deo & Trivedi, 2018) from https://arxiv.org/abs/1805.06771

## Overview

This project implements an LSTM encoder-decoder model with convolutional social pooling for predicting vehicle trajectories using the Waymo Open Motion Dataset. The model uses the motion patterns of surrounding vehicles to improve trajectory prediction accuracy in diverse traffic scenarios.

## Architecture

- **LSTM Encoder**: Processes historical trajectory sequences
- **Convolutional Social Pooling**: Captures spatial interactions between vehicles using CNN layers
- **LSTM Decoder**: Generates multi-modal trajectory predictions
- **Multi-Modal Output**: Predicts 6 different possible future trajectories with associated probabilities

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Jupyter Notebook (for data exploration)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd model-examples/selfdriving

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The model uses the **Waymo Open Motion Dataset**, which provides diverse driving scenarios with:
- Real-world vehicle trajectories from autonomous vehicle sensors
- Rich scene context including road geometry and traffic signals
- Diverse geographic locations and weather conditions

### Dataset Setup

1. Download the Waymo Open Motion Dataset:
   - Visit: https://waymo.com/open/data/motion/
   - Follow the registration and download instructions
   - Download the training and validation TFRecord files

2. Place the downloaded TFRecord files in the `waymo/` directory:
   ```
   waymo/
   ├── training/
   │   └── *.tfrecord
   └── validation/
       └── *.tfrecord
   ```

## Data Exploration

Use the included Jupyter notebook to explore and visualize the Waymo dataset:

```bash
jupyter notebook visualization/waymo_dataset_explorer.ipynb
```

The notebook provides:
- Interactive visualization of vehicle trajectories
- Scene context rendering (lanes, crosswalks, traffic signals)
- Statistical analysis of the dataset
- Sample data preprocessing pipelines

## Training

```bash
python train.py \
    --data_dir waymo/training \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --encoder_dim 128 \
    --num_modes 6
```

### Training Arguments

- `--data_dir`: Path to Waymo training TFRecord files (required)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--encoder_dim`: Hidden dimension for encoder (default: 128)
- `--num_modes`: Number of prediction modes (default: 6)
- `--hist_len`: Length of history trajectory in timesteps (default: 11, ~1.1 seconds)
- `--pred_len`: Length of predicted trajectory in timesteps (default: 80, ~8 seconds)
- `--save_dir`: Directory to save model checkpoints (default: selfdriving/checkpoints)
- `--log_dir`: Directory for tensorboard logs (default: selfdriving/logs)
- `--resume`: Path to checkpoint to resume training from
- `--num_workers`: Number of data loading workers (default: 4)
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
    --data_dir waymo/validation \
    --visualize \
    --num_samples 10
```

### Evaluation Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--data_dir`: Path to Waymo validation TFRecord files (required)
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

### 4. Waymo Dataset (`data/waymo_dataset.py`)
- Loads and preprocesses Waymo Open Motion Dataset
- Extracts vehicle trajectories and scene context
- Handles neighbor vehicle relationships
- Provides efficient data loading with caching

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