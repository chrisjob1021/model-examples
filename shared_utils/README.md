# Shared Utils

Common utilities for model training and dataset processing across different projects.

## Overview

This package provides two main utility classes:
- **DatasetProcessor**: Robust dataset preprocessing with chunked processing, resume capability, and error handling
- **ModelTrainer**: Unified interface for training models using HuggingFace Transformers infrastructure

## Installation

This package is part of the `model-examples` project. To install in development mode:

```bash
# From the project root
pip install -e .
```

## Classes

### DatasetProcessor

A comprehensive wrapper for dataset preprocessing, validation, and persistent storage with chunked processing capabilities. Designed for handling large datasets like ImageNet-1k efficiently.

#### Key Features
- **Chunked Processing**: Process large datasets in configurable chunks to prevent memory exhaustion
- **Resume Capability**: Automatically resume from interrupted processing using saved chunk progress
- **Error Handling**: Robust UTF-8 error handling for problematic datasets
- **Memory Management**: Automatic garbage collection and memory monitoring
- **Progress Tracking**: Save intermediate results and concatenate existing chunks
- **Flexible Schema**: Support custom features schema preservation

#### Constructor Parameters

```python
DatasetProcessor(
    dataset_name: str,              # Name of dataset to load via load_dataset
    output_dir: str,                # Directory for saving processed datasets
    preprocess_fn: Callable,        # Preprocessing function to apply
    processor_name: str = None,     # Instance name for file naming
    split_limits: Dict = None,      # Max samples per split {"train": 1000, "val": 100}
    num_threads: int = None,        # Processing threads (defaults to CPU count)
    start_index: int = None,        # Start processing from specific index
    chunk_size: int = 50000,        # Samples per chunk
    batch_size: int = 500,          # Batch size for map operations
    concatenate_only: bool = False, # Only concatenate existing chunks
    features = None,                # Features schema for output dataset
    **load_dataset_kwargs           # Additional arguments for load_dataset
)
```

#### Usage Example

```python
from shared_utils import DatasetProcessor
from datasets import Features, Array3D, Value

# Define features schema
features = Features({
    "pixel_values": Array3D(shape=(3, 224, 224), dtype="float32"),
    "labels": Value(dtype="int32")
})

# Define preprocessing function
def preprocess_images(examples):
    # Your preprocessing logic here
    # Process examples["image"] -> examples["pixel_values"]
    return examples

# Create processor for ImageNet
processor = DatasetProcessor(
    dataset_name="imagenet-1k",
    output_dir="./processed_datasets",
    preprocess_fn=preprocess_images,
    processor_name="imagenet_processor",
    split_limits={"train": None, "validation": None, "test": 0},
    num_threads=4,
    chunk_size=250000,
    batch_size=200,
    features=features,
    trust_remote_code=True,
    cache_dir=None  # Don't use HF cache
)

# Process dataset (with automatic resume if interrupted)
save_paths = processor.process()

# Resume from existing chunks (useful if processing was interrupted)
processor_resume = DatasetProcessor(
    dataset_name="imagenet-1k",
    output_dir="./processed_datasets",
    preprocess_fn=preprocess_images,
    concatenate_only=True  # Only concatenate existing chunks
)
processor_resume.process()
```

### ModelTrainer

Generic wrapper for HuggingFace Trainer that handles training boilerplate and orchestrates the training process.

#### Key Features
- **Timestamped Logging**: Automatically add timestamps to logging directories
- **Custom Trainer Support**: Use custom trainer classes while maintaining unified interface
- **Flexible Configuration**: Support all HuggingFace TrainingArguments

#### Constructor Parameters

```python
ModelTrainer(
    model: torch.nn.Module,           # Model to train
    training_args: TrainingArguments, # Training configuration
    train_dataset,                    # Training dataset
    eval_dataset,                     # Evaluation dataset
    preprocess_fn: Callable = None,   # Dataset preprocessing function
    data_collator: Callable = None,   # Data collation function
    trainer_class = None,             # Custom trainer class (defaults to Trainer)
    compute_metrics: Callable = None, # Metrics computation function
    compute_loss: Callable = None,    # Custom loss computation
    callbacks = None                  # Training callbacks
)
```

#### Usage Example

```python
from shared_utils import ModelTrainer
from transformers import TrainingArguments
import torch.nn as nn

# Create your model
class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Model definition...
        
model = MyModel()

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-3,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    logging_dir="./logs",
)

# Create trainer with custom trainer class
from my_custom_trainers import CNNTrainer

trainer = ModelTrainer(
    model=model,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    trainer_class=CNNTrainer,  # Optional custom trainer
    compute_metrics=compute_metrics_fn,  # Optional metrics
)

# Run training
trainer_instance, results = trainer.run()

# Access results
print(f"Training loss: {results['train_loss']}")
print(f"Evaluation loss: {results['eval_loss']}")

# Continue training or evaluate
eval_results = trainer.evaluate()
```

## Real-World Examples

### CNN on ImageNet
See the complete implementation in `cnn/`:
- `cnn/process_imagenet.py` - Uses DatasetProcessor for ImageNet preprocessing
- `cnn/train_cnn_imagenet.py` - Uses ModelTrainer for CNN training

```bash
# Process ImageNet dataset
cd cnn
python process_imagenet.py

# Train CNN model
python train_cnn_imagenet.py
```