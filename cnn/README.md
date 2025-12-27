# CNN Model Training Pipeline

This directory contains a complete CNN training pipeline for ImageNet-1k classification, implementing the PReLU (Parametric Rectified Linear Unit) activation function from the paper:

**"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"**  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Microsoft Research, 2015  
Paper: https://arxiv.org/abs/1502.01852

The implementation includes educational features such as manual convolution/pooling operations and activation visualization tools to understand what CNNs learn.

## Directory Structure

### Core Model Implementation
- **`prelu_cnn.py`** - Main CNN architecture implementation
  - Configurable PReLU/ReLU activation functions
  - Manual convolution and pooling implementations (educational)
  - PyTorch built-in fallback options
  - Spatial pyramid pooling support
  - Custom HuggingFace trainer integration

### Data Preprocessing (Optional)
- **`process_imagenet.py`** - Preprocesses raw ImageNet-1k dataset
  - **Note: Preprocessing is completely optional** - the training script uses `torchvision.transforms` to apply transformations on-the-fly
  - Converts images to preprocessed tensors
  - Uses [shared DatasetProcessor utility](../shared_utils/dataset_processor.py)
  - Supports chunk-based processing and saving for large datasets
  - Usage: `python process_imagenet.py [--concatenate-only]`
    - `--concatenate-only`: Only concatenate existing progress chunks, don't process new data

- **`analyze_processed_dataset.py`** - Validates processed datasets
  - Only needed if using preprocessed data
  - Verifies chunk alignment with original ImageNet
  - Checks data integrity and label consistency
  - Run: `python analyze_processed_dataset.py`

### Training
- **`train_cnn_imagenet.py`** - Main training script
  - Trains CNN model on ImageNet using HuggingFace Transformers
  - Applies transforms on-the-fly using `torchvision.transforms` (no preprocessing required)
  - Comprehensive data augmentation pipeline (RandomResizedCrop, RandAugment, etc.)
  - Handles common ImageNet issues (EXIF errors, corrupted images)
  - Saves training logs to `logs/<date>_<timestamp>/` for TensorBoard visualization
  - Saves model checkpoints to `./results`
  - Run: `python train_cnn_imagenet.py`

### Evaluation & Deployment
- **`evaluate_model.py`** - Evaluates trained models on validation set
  - Calculates top-1 and top-5 accuracy percentages
  - Automatically detects and loads latest checkpoint
  - Smart activation detection (PReLU vs ReLU)
  - Progress tracking with real-time metrics
  - Run: `python evaluate_model.py`
  - Configuration: Edit the script directly to change settings like batch size or limit batches for testing

- **`upload_to_huggingface.py`** - Upload trained models to HuggingFace Hub
  - Creates comprehensive model cards with training details
  - Auto-detects latest checkpoint and activation type
  - Includes model weights, config, and metadata
  - Supports organization uploads and private repos
  - Run: `python upload_to_huggingface.py --repo-name MODEL_NAME`
  - See workflow section for detailed usage examples

### Analysis & Visualization
- **`visualize_activations.py`** - Visualizes learned features
  - Shows activation maps for each CNN layer
  - Helps understand what different layers learn
  - Loads trained models and analyzes internal representations
  - Run: `python visualize_activations.py`

### Testing & Utilities
- **`test_cuda.py`** - Tests CUDA/GPU functionality
  - Verifies proper device placement
  - Tests manual layer implementations
  - Ensures GPU utilization is working correctly
  - Run: `python test_cuda.py`

### Jupyter Notebooks
- **`prelu_cnn.ipynb`** - Interactive CNN development and testing
  - Demonstrates data loading and preprocessing
  - Tests grayscale to RGB conversion
  - Validates data pipeline with shuffling experiments

- **`prelu_cnn_processed.ipynb`** - Works with preprocessed datasets
  - Uses pre-processed ImageNet data (if available) mentioned above

- **`toy_cnn.ipynb`** - Simplified CNN experiments
  - Educational notebook for learning CNN concepts

## Setup

### Environment Setup
```bash
# Set up Python virtual environment with all dependencies
scripts/setup_venv.sh

# The setup script will:
# - Create a Python virtual environment
# - Install all required packages from requirements.txt
# - Activate the environment
```

## Workflow

### 0. Data Preparation (Optional)
```bash
# Option A: Use raw ImageNet directly (recommended for flexibility) through the training script
# The training script will apply transforms on-the-fly using torchvision.transforms
# No preprocessing needed - skip to step 2!

# Option B: Preprocess dataset
python process_imagenet.py
```

### 1. Model Training
```bash
# Train CNN model
python train_cnn_imagenet.py

# Training logs are saved to logs/<date>_<timestamp>/
```

### 2. Monitor Training
```bash
# Start TensorBoard to visualize training progress
scripts/start_tensorboard.sh

# The script automatically:
# - Finds and displays available log directories
# - Starts TensorBoard on port 6006
# - Opens http://localhost:6006 in your browser
```

### 3. Evaluate Model
```bash
# Evaluate the trained model on validation set
python evaluate_model.py

# To test with limited batches or change settings, edit the configuration
# variables at the top of the main() function in evaluate_model.py
```

### 4. Upload to HuggingFace Hub
```bash
# First, authenticate with HuggingFace
huggingface-cli login

# Basic upload (auto-detects latest checkpoint)
python upload_to_huggingface.py --repo-name cnn-prelu-imagenet

# Upload with accuracy metrics
python upload_to_huggingface.py \
  --repo-name cnn-prelu-imagenet \
  --top1-acc 76.5 \
  --top5-acc 93.2

# Upload specific checkpoint
python upload_to_huggingface.py \
  --repo-name cnn-prelu-imagenet \
  --checkpoint results/cnn_results_prelu/checkpoint-375300 \
  --top1-acc 76.5 \
  --top5-acc 93.2

# Upload to organization (private)
python upload_to_huggingface.py \
  --repo-name cnn-prelu-imagenet \
  --organization your-org \
  --private \
  --top1-acc 76.5 \
  --top5-acc 93.2
```

**Upload Options:**
- `--repo-name`: Required. Name for the HuggingFace repository
- `--checkpoint`: Optional. Path to checkpoint (defaults to latest)
- `--top1-acc`: Optional. Top-1 accuracy for model card
- `--top5-acc`: Optional. Top-5 accuracy for model card
- `--organization`: Optional. Upload to HF organization
- `--private`: Optional. Make repository private

The upload includes:
- Model weights (`model.safetensors`)
- Configuration (`config.json`)
- Comprehensive model card (`README.md`)
- Training metadata (`trainer_state.json`)

## Key Features

- **PReLU Implementation**: Parametric ReLU as proposed in the original paper for improved accuracy
- **Educational Value**: Manual implementations alongside production-ready code
- **Flexible Non-Linearity**: Compare PReLU vs ReLU performance
- **Robust Processing**: Handles common ImageNet dataset issues
- **Modern Training**: Uses HuggingFace Transformers infrastructure
- **Comprehensive Visualization**: Understand what CNNs learn at each layer
- **Model Evaluation**: Evaluation script with top-1 and top-5 accuracy metrics
- **HuggingFace Hub Integration**: One-command upload with auto-generated model cards

## Requirements

See `requirements.txt` in the parent directory for dependencies. Key packages:
- PyTorch
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- Pillow (PIL)
- NumPy
- Matplotlib (for visualization)

## Notes

- The manual convolution/pooling implementations are for educational purposes and will be slower than PyTorch's optimized versions
- Set `use_builtin_conv=True` in the CNN model for production use
- The ImageNet-1k dataset requires significant storage space
- GPU is highly recommended for training (CUDA-compatible)