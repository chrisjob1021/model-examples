"""Minimal CNN training example using ImageNet

Convolution and pooling are implemented manually with explicit loops instead of
relying on ``torch.nn.functional`` helpers such as ``conv2d`` or ``max_pool2d``.
This file trains the network on ImageNet with either ReLU or PReLU activation so
you can compare the impact of PReLU on accuracy.
"""

# Avoid using einsum, conv2d or max_pool2d as requested

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer
import numpy as np
from PIL import Image
import math
import warnings

class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_prelu=False, use_builtin_conv=False, prelu_channel_wise=True):
        super().__init__()
        self.conv = ManualConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, use_builtin=use_builtin_conv)
        
        # Create appropriate activation function
        if use_prelu:
            # Channel-wise: one parameter per output channel
            # Channel-shared: one parameter for all channels
            self.act = nn.PReLU(out_channels if prelu_channel_wise else 1)
        else:
            self.act = nn.ReLU(inplace=True)

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
        
        # Warn if using slow manual implementation
        if not use_builtin:
            warnings.warn(
                "ManualConv2d is using slow manual loop implementation. "
                "Set use_builtin=True for faster PyTorch F.conv2d implementation.",
                UserWarning,
                stacklevel=2
            )

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

    def __init__(self, kernel_size, stride=None, use_builtin=False, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        # Default stride equals kernel size as in typical max pooling layers
        self.stride = stride or kernel_size
        self.use_builtin = use_builtin
        self.padding = padding
        
        # Warn if using slow manual implementation
        if not use_builtin:
            warnings.warn(
                "ManualMaxPool2d is using slow manual loop implementation. "
                "Set use_builtin=True for faster PyTorch F.max_pool2d implementation.",
                UserWarning,
                stacklevel=2
            )

    def forward(self, x):
        if self.use_builtin:
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        
        # Add padding to input if specified
        if self.padding > 0:
            # For max pooling, padding is typically done with -inf or very negative values
            # so they don't interfere with the max operation
            x = F.pad(x, (self.padding,) * 4, value=float('-inf'))
        
        # x shape: (batch, channels, height, width)
        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

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
                            row_start : row_start + self.kernel_size,
                            col_start : col_start + self.kernel_size,
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
        
        # Warn if using slow manual implementation
        if not use_builtin:
            warnings.warn(
                "ManualAdaptiveAvgPool2d is using slow manual loop implementation. "
                "Set use_builtin=True for faster PyTorch F.adaptive_avg_pool2d implementation.",
                UserWarning,
                stacklevel=2
            )

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

class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer that pools at multiple scales."""
    
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels
    
    def forward(self, x):
        pyramid_features = []
            
        for level in self.levels:
            # F.adaptive_avg_pool2d(x, (level, level)) produces shape: (batch, channels, level, level)
            pooled = F.adaptive_avg_pool2d(x, (level, level))
            # flatten(2) keeps dims 0,1 and flattens the rest, so shape becomes: (batch, channels, level*level)
            pyramid_features.append(pooled.flatten(2))

        # Before cat: pyramid_features contains tensors of shapes:
        # - level 1: (batch, channels, 1) 
        # - level 2: (batch, channels, 4)
        # - level 4: (batch, channels, 16)
        # torch.cat(pyramid_features, dim=2) concatenates along the flattened spatial dimension
        # Final shape: (batch, channels, 1+4+16=21)
        return torch.cat(pyramid_features, dim=2)

# -------------------------------------------------------
# Simple CNN using the manual layers
# -------------------------------------------------------
class CNN(nn.Module):
    """CNN architecture based on the PReLU paper for ImageNet classification.
    
    This implements the exact architecture from "Delving Deep into Rectifiers" 
    by He et al. (2015) to recreate their ImageNet experimental results.
    
    Parameters
    ----------
    use_prelu : bool, optional
        If ``True``, ``nn.PReLU`` is used for activations instead of ``nn.ReLU``.
    use_builtin_conv : bool, optional
        Forward ``True`` to use the faster PyTorch convolution implementation
        inside :class:`ManualConv2d`. Defaults to ``False``.
    prelu_channel_wise : bool, optional
        If ``True``, use channel-wise PReLU (one parameter per channel).
        If ``False``, use channel-shared PReLU (one parameter per layer).
        Only used when use_prelu=True. Defaults to ``True``.
    """

    def __init__(self, use_prelu: bool = False, *, use_builtin_conv: bool = True, prelu_channel_wise: bool = True, num_classes: int = 1000): 
        super().__init__()
        self.num_classes = num_classes
        
        # Warn if using slow manual convolutions
        if not use_builtin_conv:
            warnings.warn(
                "CNN is using slow manual convolution implementations. "
                "Set use_builtin_conv=True for faster training with PyTorch's built-in convolutions.",
                UserWarning,
                stacklevel=2
            )
        
        # Layer 1: 7×7 conv, 64 filters, stride=2 (ImageNet input: 224×224)
        self.conv1 = nn.Sequential(
            ConvAct(3, 64, kernel_size=7, stride=2, padding=3, use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise),  # 224×224 → 112×112
            ManualMaxPool2d(kernel_size=3, stride=3, padding=0, use_builtin=use_builtin_conv)  # 112×112 → 37×37
        )
        
        # conv2_x: 4 layers of 2×2, 128 filters
        self.conv2 = nn.Sequential(
            *[ConvAct(64 if i == 0 else 128, 128, kernel_size=2, stride=1, padding=0, use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise) for i in range(4)],  # 37×37 → 36×36 → 35×35 → 34×34 → 33×33
            ManualMaxPool2d(kernel_size=2, stride=2, use_builtin=use_builtin_conv)  # 33×33 → 16×16
        )
        
        # conv3_x: 6 layers of 2×2, 256 filters  
        self.conv3 = nn.Sequential(
            *[ConvAct(128 if i == 0 else 256, 256, kernel_size=2, stride=1, padding=0, use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise) for i in range(6)],  # 16×16 → 15×15 → 14×14 → 13×13 → 12×12 → 11×11
        )
        
        # Spatial Pyramid Pooling (as used in PReLU paper)
        # Levels {6,3,2,1} create 6²+3²+2²+1² = 36+9+4+1 = 50 bins per channel
        self.spp = nn.Sequential(
            SpatialPyramidPooling(levels=[6, 3, 2, 1]),
            nn.Flatten(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 50, 4096, bias=True),               # 12 800 → 4 096
            nn.PReLU(4096 if (use_prelu and prelu_channel_wise) else 1)
                if use_prelu else nn.ReLU(inplace=True),
            nn.Dropout(0.5),                                    # ← after fc1 activation

            nn.Linear(4096, 4096, bias=True),                   # 4 096 → 4 096
            nn.PReLU(4096 if (use_prelu and prelu_channel_wise) else 1)
                if use_prelu else nn.ReLU(inplace=True),
            nn.Dropout(0.5),                                    # ← after fc2 activation

            nn.Linear(4096, num_classes)                        # 4 096 → 1 000 (or 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply SPP and flatten
        x = self.spp(x)  # (batch, 256, 50)
        
        return self.classifier(x)

# -------------------------------------------------------
# Data preprocessing functions
# -------------------------------------------------------
def preprocess_images(examples):
    """Preprocess images for ImageNet."""
    images = []
    
    # ImageNet normalization values (standard for pretrained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for image in examples['image']:  # ImageNet uses 'image' column
        if isinstance(image, str):
            # If image is a file path, load it and convert to RGB color mode (3 channels: Red, Green, Blue)
            # This ensures consistent color format regardless of input image type
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If image is numpy array, convert to PIL Image (Python Imaging Library)
            # Handle both grayscale and RGB images by converting to RGB
            if len(image.shape) == 2:
                # Grayscale image - convert to RGB
                image = Image.fromarray(image).convert('RGB')
            elif len(image.shape) == 3:
                # RGB image
                image = Image.fromarray(image)
            else:
                # Unexpected format - convert to RGB
                image = Image.fromarray(image).convert('RGB')
        
        # Resize to 224×224 (ImageNet standard)
        image = image.resize((224, 224), Image.BILINEAR)
        
        # Convert to tensor - at this point image is always RGB (3D HWC)
        image = torch.tensor(np.array(image), dtype=torch.float32)
        
        # Rearrange dimensions from Height-Width-Channel (HWC) to Channel-Height-Width (CHW)
        # This is required because PyTorch expects images in CHW format, but PIL/OpenCV use HWC
        image = image.permute(2, 0, 1)
        
        # Normalize pixel values from [0, 255] range to [0, 1] range
        # Neural networks work better with normalized inputs, and 255.0 ensures float division
        image = image / 255.0
        for c in range(3):
            image[c] = (image[c] - mean[c]) / std[c]
        # After normalization: values typically range from ~[-2.5, 2.5]
        # This is because: (0 - mean) / std ≈ -2.1 and (1 - mean) / std ≈ 2.6
            
        images.append(image)
    
    examples['pixel_values'] = images
    examples['labels'] = examples['label']  # ImageNet uses 'label' column
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
