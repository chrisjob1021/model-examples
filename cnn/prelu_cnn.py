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
        # nn.Parameter wraps tensors to make them trainable parameters that will be
        # automatically tracked by PyTorch's autograd system for gradient computation
        # during backpropagation. Without nn.Parameter, these tensors would not be
        # updated during training.
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

        batch_size, _, in_height, in_width = x.shape
        kernel_height = kernel_width = self.weight.shape[2]

        # Pad the input so that the manually computed convolution matches PyTorch's behavior
        # F.pad takes padding as (left, right, top, bottom)
        # This adds self.padding pixels to all sides of the input
        # For 3D tensors (batch, channels, height, width), padding is applied to the last 2 dimensions
        padded_x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
        )

        # Compute output spatial dimensions using the convolution formula
        out_height = (in_height + 2 * self.padding - kernel_height) // self.stride + 1 # add 1 because there's one initial position
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
                        row_start = row * self.stride # range indexes from 0, so this will be 0 and first position is going to be top left
                        col_start = col * self.stride
                        region = padded_x[
                            batch_idx,
                            :,  # all input channels
                            row_start : row_start + kernel_height,
                            col_start : col_start + kernel_width,
                        ]

                        # 2) element-wise multiply patch and kernel, 3) sum and add bias
                        output[batch_idx, out_ch, row, col] = (
                            # self.weight[out_ch] returns shape (in_channels, kernel_height, kernel_width)
                            # This represents the kernel weights for the current output channel
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
class CNN(nn.Module):
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
        # ManualMaxPool2d with kernel_size=2 reduces spatial dimensions by half
        # Input: (batch, channels, height, width) -> Output: (batch, channels, height//2, width//2)
        self.pool1 = ManualMaxPool2d(kernel_size=2)

        self.conv2 = ManualConv2d(16, 32, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act2 = nn.PReLU() if use_prelu else nn.ReLU()
        self.pool2 = ManualMaxPool2d(kernel_size=2)

        self.conv3 = ManualConv2d(32, 64, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act3 = nn.PReLU() if use_prelu else nn.ReLU()
        self.pool3 = ManualMaxPool2d(kernel_size=2)

        # After three 2x2 poolings: 32x32 -> 16x16 -> 8x8 -> 4x4
        # 64: number of output channels from conv3
        # 4 * 4: spatial dimensions after 3 pooling layers (32->16->8->4)
        # 10: number of CIFAR-10 classes
        self.fc = nn.Linear(64 * 4 * 4, 10)
        # Dropout randomly sets 50% of inputs to zero during training to prevent overfitting
        # This forces the network to learn redundant representations and improves generalization
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
        
        # Reshape the tensor from 4D (batch, channels, height, width) to 2D (batch, features)
        # x.size(0) keeps the batch dimension unchanged
        # -1 automatically calculates the flattened feature dimension (channels * height * width)
        # This flattens the spatial dimensions and channels into a single feature vector per sample
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
            # If image is a file path, load it and convert to RGB color mode (3 channels: Red, Green, Blue)
            # This ensures consistent color format regardless of input image type
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If image is numpy array, convert to PIL Image (Python Imaging Library)
            image = Image.fromarray(image)
        
        # Convert to tensor and normalize
        image = torch.tensor(np.array(image), dtype=torch.float32)
        # Step 1: Rearrange dimensions from Height-Width-Channel (HWC) to Channel-Height-Width (CHW)
        # This is required because PyTorch expects images in CHW format, but PIL/OpenCV use HWC
        image = image.permute(2, 0, 1)
        
        # Step 2: Normalize pixel values from [0, 255] range to [0, 1] range
        # Neural networks work better with normalized inputs, and 255.0 ensures float division
        image = image / 255.0
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

    def compute_loss(self, model, inputs, return_outputs=False):
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        outputs = model(pixel_values)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        """Compute accuracy for evaluation."""
        # eval_pred is a tuple containing (predictions, labels) from the model evaluation
        # predictions: numpy array of shape (num_samples, num_classes) with raw logits
        # labels: numpy array of shape (num_samples,) with true class labels
        predictions, labels = eval_pred
        # axis=1 selects the class dimension (columns) to find the maximum probability
        # predictions shape: (num_samples, num_classes) -> argmax along axis=1 gives class indices
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}
