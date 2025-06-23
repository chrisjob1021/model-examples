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

## Features

- **Automatic preprocessing**: Apply preprocessing functions to datasets
- **Custom trainer support**: Use custom trainer classes
- **Timestamped logging**: Prevent log overwrites with automatic timestamps
- **Unified interface**: Consistent training interface across different models

## Examples

See the `cnn/` directory for examples using this package with CNN models. 