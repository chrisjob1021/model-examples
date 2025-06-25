# Shared Utils

Common utilities for model training across different projects.

## Installation

This package is part of the `model-examples` project. To install in development mode:

```bash
# From the project root
pip install -e .
```

## Usage

### ModelTrainer

The `ModelTrainer` class provides a unified interface for training models using Hugging Face's Trainer infrastructure.

```python
from shared_utils import ModelTrainer
from transformers import TrainingArguments

# Create your model
model = YourModel()

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # ... other args
)

# Create trainer
trainer = ModelTrainer(
    model=model,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    preprocess_fn=your_preprocess_function,  # optional
    trainer_class=YourCustomTrainer,  # optional, defaults to transformers.Trainer
)

# Run training
trainer_instance, results = trainer.run()
```

### DatasetProcessor

The `DatasetProcessor` class provides a unified interface for dataset preprocessing, validation, and persistent storage.

```python
from shared_utils import DatasetProcessor
from datasets import Dataset

# Define preprocessing function
def preprocess_function(examples):
    # Your preprocessing logic here
    return examples

# Define validation function (optional)
def validate_dataset(dataset):
    # Your validation logic here
    return True

# Create processor
processor = DatasetProcessor(
    dataset=your_dataset,
    output_dir="./processed_data",
    processor_name="my_processor",
    preprocess_fn=preprocess_function,
    validation_fn=validate_dataset,  # optional
    save_format="parquet",  # "auto", "parquet", "json", "pickle", "csv"
    metadata={"source": "original_dataset", "version": "1.0"}  # optional
)

# Process and save dataset
results = processor.process()

# Or get processed dataset without saving
processed_dataset = processor.get_processed_dataset()
```

#### Supported Input Formats

- HuggingFace `Dataset` or `DatasetDict`
- Pandas `DataFrame`
- Python dictionary

#### Supported Output Formats

- **Parquet** (default): Fast, compressed, good for large datasets
- **JSON**: Human-readable, good for smaller datasets
- **CSV**: Simple tabular format
- **Pickle**: Python-specific serialization

#### Features

- **Automatic preprocessing**: Apply custom preprocessing functions
- **Dataset validation**: Validate processed datasets with custom functions
- **Multiple output formats**: Save in various formats with auto-detection
- **Metadata tracking**: Save processing metadata and statistics
- **Timestamped outputs**: Prevent file overwrites with automatic timestamps
- **Logging**: Comprehensive logging of processing steps
- **Flexible input**: Support for various dataset types

## Features

- **Automatic preprocessing**: Apply preprocessing functions to datasets
- **Custom trainer support**: Use custom trainer classes
- **Timestamped logging**: Prevent log overwrites with automatic timestamps
- **Unified interface**: Consistent training interface across different models
- **Dataset processing**: Comprehensive dataset preprocessing and storage utilities

## Examples

See the `cnn/` directory for examples using this package with CNN models. 