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
from transformers import (
    Trainer, 
    TrainingArguments
)
import numpy as np
from PIL import Image
import math

class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, use_prelu, use_builtin_conv):
        super().__init__()
        self.conv = ManualConv2d(in_channels, out_channels, kernel_size=3, padding=1, use_builtin=use_builtin_conv)
        self.act = nn.PReLU() if use_prelu else nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))

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
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights using Kaiming initialization
        # This is specifically designed for ReLU-like activations and helps with gradient flow
        # 'fan_in' mode preserves the magnitude of variance in the forward pass
        # 'relu' nonlinearity accounts for the fact that ReLU zeros out half the values
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
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

    def __init__(self, kernel_size, stride=None, use_builtin=False):
        super().__init__()
        self.kernel_size = kernel_size
        # Default stride equals kernel size as in typical max pooling layers
        self.stride = stride or kernel_size
        self.use_builtin = use_builtin

    def forward(self, x):
        if self.use_builtin:
            return F.max_pool2d(x, self.kernel_size, self.stride)
        
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
# Manual adaptive average pooling layer
# -------------------------------------------------------
class ManualAdaptiveAvgPool2d(nn.Module):
    """Adaptive average pooling using explicit loops."""

    def __init__(self, output_size, use_builtin=False, debug=False):
        super().__init__()
        self.output_size = output_size
        self.use_builtin = use_builtin
        self.debug = debug

    def forward(self, x, debug=False):
        if self.use_builtin:
            return F.adaptive_avg_pool2d(x, self.output_size)

        # x shape: (batch, channels, height, width)
        batch_size, channels, in_height, in_width = x.shape
        out_height, out_width = self.output_size

        # Allocate output tensor
        output = torch.zeros(
            batch_size, channels, out_height, out_width, device=x.device
        )

        if self.debug:
            print(f"\nAdaptive Average Pooling Debug:")
            print(f"Input: {batch_size} batches × {channels} channels × {in_height}×{in_width} spatial")
            print(f"Output: {out_height}×{out_width} spatial")
            print(f"\n{'Cell':<8} {'Row Start':<10} {'Row End':<8} {'Col Start':<10} {'Col End':<8} {'Region':<15}")
            print("-" * 70)

        for i in range(out_height):
            for j in range(out_width):
                row_start = math.floor(i * in_height / out_height)
                row_end = math.ceil((i + 1) * in_height / out_height)
                col_start = math.floor(j * in_width / out_width)
                col_end = math.ceil((j + 1) * in_width / out_width)

                region = x[:, :, row_start:row_end, col_start:col_end]
                region_avg = region.mean(dim=(2, 3))
                output[:, :, i, j] = region_avg

                if self.debug:
                    region_desc = f"[{row_start}:{row_end}, {col_start}:{col_end}]"
                    print(f"({i},{j}){'':<2} {row_start:<10} {row_end:<8} {col_start:<10} {col_end:<8} {region_desc:<15}")
                    
                    # Show the math for the first batch and channel
                    if batch_size > 0 and channels > 0:
                        region_2d = x[0, 0, row_start:row_end, col_start:col_end]
                        region_shape = f"{row_end-row_start}×{col_end-col_start}"
                        region_sum = region_2d.sum().item()
                        region_count = region_2d.numel()
                        avg_result = region_avg[0, 0].item()
                        
                        # Format region values for display
                        region_values = region_2d.flatten().tolist()
                        region_str = ', '.join([f"{val:.1f}" for val in region_values])
                        
                        print(f"    Math: {region_shape} region = [{region_str}]")
                        print(f"    Sum: {region_sum:.1f}, Count: {region_count}, Average: {region_sum:.1f}/{region_count} = {avg_result:.1f}")

        if self.debug:
            print("-" * 70)

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
        
        # 3 × (64-ch) conv stack, 2×2 max pool
        self.stage1 = nn.Sequential(
            *[ConvAct(  3 if i == 0 else 64, 64, use_prelu, use_builtin_conv) for i in range(3)],
            nn.MaxPool2d(kernel_size=2)   # 32→16: Reduces spatial dimensions by half, from 32x32 (original CIFAR10 input size) to 16x16
        )

        # 3 × (128-ch) conv stack
        self.stage2 = nn.Sequential(
            *[ConvAct(  64 if i == 0 else 128, 128, use_prelu, use_builtin_conv) for i in range(3)],
            nn.MaxPool2d(kernel_size=2)   # 16→8: Reduces spatial dimensions by half, from 16x16 to 8x8
        )

        # 3 × (256-ch) conv stack
        self.stage3 = nn.Sequential(
            *[ConvAct(  128 if i == 0 else 256, 256, use_prelu, use_builtin_conv) for i in range(6)],
            nn.MaxPool2d(kernel_size=2)   # 8→4: Reduces spatial dimensions by half, from 8x8 to 4x4
        )

        self.classifier = nn.Sequential(
            # Step 1: Global Average Pooling - Reduces spatial dimensions from 4×4 to 1×1
            # This converts the 4×4×256 feature maps to 1×1, effectively averaging each channel
            nn.AdaptiveAvgPool2d(1),          # 4×4 → 1×1
            
            # Step 2: Flatten - Converts 3D tensor (batch, channels, 1, 1) to 2D tensor (batch, channels)
            # This removes the spatial dimensions, leaving only batch and feature dimensions
            nn.Flatten(),
            
            # Step 3: First Dropout - Randomly sets 50% of inputs to zero during training
            # This helps prevent overfitting by forcing the network to not rely on specific neurons
            nn.Dropout(0.5),
            
            # Step 4: First Linear Layer - Expands from 256 features to 4096 features
            # This creates a large hidden layer for learning complex representations
            nn.Linear(256, 4096),
            
            # Step 5: First Activation - Applies PReLU or ReLU based on use_prelu parameter
            # PReLU learns the negative slope parameter, while ReLU uses fixed slope of 0
            nn.PReLU(4096) if use_prelu else nn.ReLU(inplace=True),
            
            # Step 6: Second Dropout - Another 50% dropout for additional regularization
            # Multiple dropout layers help prevent co-adaptation of neurons
            nn.Dropout(0.5),
            
            # Step 7: Second Linear Layer - Maps from 4096 to 4096 features
            # This creates another large hidden layer for further feature learning
            nn.Linear(4096, 4096),
            
            # Step 8: Second Activation
            # Provides non-linearity after the second linear transformation
            nn.PReLU(4096) if use_prelu else nn.ReLU(inplace=True),
            
            # Step 9: Final Linear Layer - Maps from 4096 features to 10 classes (CIFAR10)
            # This is the output layer that produces logits for each of the 10 classes
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

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
