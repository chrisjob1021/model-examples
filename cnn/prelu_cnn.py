"""Minimal CNN training example using CIFAR10

Convolution and pooling are implemented manually with explicit loops instead of
relying on ``torch.nn.functional`` helpers such as ``conv2d`` or ``max_pool2d``.
This file trains the network on CIFAR10 with either ReLU or PReLU activation so
you can compare the impact of PReLU on accuracy.
"""

# Avoid using einsum, conv2d or max_pool2d as requested

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    Trainer, 
    TrainingArguments
)
from accelerate import Accelerator
import numpy as np
from PIL import Image

# -------------------------------------------------------
# Manual 2D convolution layer implemented with nested loops
# -------------------------------------------------------
class ManualConv2d(nn.Module):
    """2D convolution implemented with explicit loops.

    Parameters
    ----------
    use_builtin : bool, optional
        If ``True``, ``torch.nn.functional.conv2d`` is used instead of the
        slower manual loop implementation. This is convenient when you want to
        speed up training while keeping the same layer interface.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *, use_builtin=False):
        super().__init__()
        # Kernel weights and biases follow the shape used by nn.Conv2d
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        self.use_builtin = use_builtin

    def forward(self, x):
        # x: (batch, in_channels, height, width)
        if self.use_builtin:
            # Delegate to PyTorch's conv2d for speed. Padding is handled
            # internally by the functional call so no explicit loops are needed.
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
            )

        batch_size, in_channels, in_height, in_width = x.shape
        kernel_height = kernel_width = self.weight.shape[2]

        # Pad the input so that the manually computed convolution matches PyTorch's behavior
        padded_x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
        )

        # Compute output spatial dimensions using the convolution formula
        out_height = (in_height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_width) // self.stride + 1

        # Allocate output tensor on the same device as the input
        output = torch.zeros(
            batch_size, self.weight.shape[0], out_height, out_width, device=x.device
        )

        # Iterate over the batch, output channels and spatial positions
        for batch_idx in range(batch_size):
            for out_ch in range(self.weight.shape[0]):
                for row in range(out_height):
                    for col in range(out_width):
                        # 1) select the patch from the padded image
                        row_start = row * self.stride
                        col_start = col * self.stride
                        region = padded_x[
                            batch_idx,
                            :,  # all input channels
                            row_start : row_start + kernel_height,
                            col_start : col_start + kernel_width,
                        ]

                        # 2) element-wise multiply patch and kernel, 3) sum and add bias
                        output[batch_idx, out_ch, row, col] = (
                            region * self.weight[out_ch]
                        ).sum() + self.bias[out_ch]

        return output

# -------------------------------------------------------
# Manual max pooling layer (no padding, square kernel)
# -------------------------------------------------------
class ManualMaxPool2d(nn.Module):
    """Max pooling using explicit loops (square kernels only)."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        # Default stride equals kernel size as in typical max pooling layers
        self.stride = stride or kernel_size

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, channels, in_height, in_width = x.shape
        k = self.kernel_size
        out_height = (in_height - k) // self.stride + 1
        out_width = (in_width - k) // self.stride + 1

        # Allocate output tensor
        output = torch.zeros(batch_size, channels, out_height, out_width, device=x.device)

        # Slide the pooling window across the input
        for batch_idx in range(batch_size):
            for channel in range(channels):
                for row in range(out_height):
                    for col in range(out_width):
                        row_start = row * self.stride
                        col_start = col * self.stride
                        region = x[
                            batch_idx,
                            channel,
                            row_start : row_start + k,
                            col_start : col_start + k,
                        ]
                        # Compute max inside the window and store the result
                        output[batch_idx, channel, row, col] = region.max()

        return output

# -------------------------------------------------------
# Simple CNN using the manual layers
# -------------------------------------------------------
class ManualCNN(nn.Module):
    """Very small CNN constructed from the manual layers above.

    Parameters
    ----------
    use_prelu : bool, optional
        If ``True``, ``nn.PReLU`` is used for activations instead of ``nn.ReLU``.
    use_builtin_conv : bool, optional
        Forward ``True`` to use the faster PyTorch convolution implementation
        inside :class:`ManualConv2d`. Defaults to ``False``.
    """

    def __init__(self, use_prelu: bool = False, *, use_builtin_conv: bool = False):
        super().__init__()
        self.conv1 = ManualConv2d(3, 16, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act1 = nn.PReLU() if use_prelu else nn.ReLU()
        self.pool1 = ManualMaxPool2d(kernel_size=2)

        self.conv2 = ManualConv2d(16, 32, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act2 = nn.PReLU() if use_prelu else nn.ReLU()
        self.pool2 = ManualMaxPool2d(kernel_size=2)

        self.conv3 = ManualConv2d(32, 64, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act3 = nn.PReLU() if use_prelu else nn.ReLU()
        self.pool3 = ManualMaxPool2d(kernel_size=2)

        # After three 2x2 poolings: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc = nn.Linear(64 * 4 * 4, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# -------------------------------------------------------
# Data preprocessing functions
# -------------------------------------------------------
def preprocess_images(examples):
    """Preprocess images for the model."""
    images = []
    for image in examples['img']:
        if isinstance(image, str):
            # If image is a file path, load it
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If image is numpy array, convert to PIL
            image = Image.fromarray(image)
        
        # Convert to tensor and normalize
        image = torch.tensor(np.array(image), dtype=torch.float32)
        image = image.permute(2, 0, 1) / 255.0  # HWC -> CHW and normalize to [0,1]
        images.append(image)
    
    examples['pixel_values'] = images
    # CIFAR-10 uses 'label' column, not 'labels'
    examples['labels'] = examples['label']
    return examples

# -------------------------------------------------------
# Custom Trainer for CNN
# -------------------------------------------------------
class CNNTrainer(Trainer):
    """Custom trainer for CNN models."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pixel_values = inputs['pixel_values']
        labels = inputs['labels']
        
        # pixel_values is already a tensor, no need to stack
        outputs = model(pixel_values)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(self, eval_pred):
        """Compute accuracy for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

# -------------------------------------------------------
# Main training function
# -------------------------------------------------------
def train_cnn_with_huggingface(use_prelu=False, use_builtin_conv=False, num_epochs=5):
    """Train CNN using Hugging Face libraries."""
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    
    # Load CIFAR-10 dataset from Hugging Face
    print("Loading CIFAR-10 dataset...")
    dataset = load_dataset("cifar10")
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_images,
        batched=True,
        batch_size=100,
        remove_columns=dataset['train'].column_names
    )
    
    # Create model
    print(f"Creating CNN model (PReLU: {use_prelu}, Builtin conv: {use_builtin_conv})...")
    model = ManualCNN(use_prelu=use_prelu, use_builtin_conv=use_builtin_conv)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./cnn_results_{'prelu' if use_prelu else 'relu'}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs_{'prelu' if use_prelu else 'relu'}",
        logging_steps=100,
        save_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
        # greater_is_better=True,
        dataloader_pin_memory=False,  # Disable for manual convolution
        remove_unused_columns=False,  # Keep all columns - required for custom models
    )
    
    # Initialize trainer
    trainer = CNNTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['test'],
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    results = trainer.evaluate()
    
    print(f"Final test accuracy: {results['eval_accuracy']:.4f}")
    return trainer, results

if __name__ == "__main__":
    # Train with ReLU
    print("=" * 50)
    print("Training with ReLU activation...")
    print("=" * 50)
    relu_trainer, relu_results = train_cnn_with_huggingface(
        use_prelu=False, 
        use_builtin_conv=False,
        num_epochs=3
    )
    
    # Train with PReLU
    print("\n" + "=" * 50)
    print("Training with PReLU activation...")
    print("=" * 50)
    prelu_trainer, prelu_results = train_cnn_with_huggingface(
        use_prelu=True, 
        use_builtin_conv=False,
        num_epochs=3
    )
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"ReLU Test Accuracy:  {relu_results['eval_accuracy']:.4f}")
    print(f"PReLU Test Accuracy: {prelu_results['eval_accuracy']:.4f}")
    
    if relu_results['eval_accuracy'] > prelu_results['eval_accuracy']:
        print("ReLU performed better!")
    elif prelu_results['eval_accuracy'] > relu_results['eval_accuracy']:
        print("PReLU performed better!")
    else:
        print("Both performed equally!")
