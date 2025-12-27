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
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional
from timm.layers import DropPath  # Stochastic depth implementation
from torch.utils.checkpoint import checkpoint  # Gradient checkpointing

class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_prelu=False, use_builtin_conv=False, prelu_channel_wise=True, bn_momentum=0.1):
        super().__init__()
        self.conv = ManualConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, use_builtin=use_builtin_conv)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        # Create appropriate activation function
        if use_prelu:
            # Channel-wise: one parameter per output channel
            # Channel-shared: one parameter for all channels
            self.act = nn.PReLU(out_channels if prelu_channel_wise else 1)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_prelu=False, use_builtin_conv=False, prelu_channel_wise=True, bn_momentum=0.1):
        super().__init__()
        # Calculate padding to maintain spatial dimensions
        # Convolution output size formula:
        #   output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
        #
        # For same padding (output_size = input_size) with stride=1:
        # Let's call input_size = S for clarity
        #   S = floor((S + 2*padding - kernel_size) / 1) + 1
        #   S = S + 2*padding - kernel_size + 1
        #   S - S = 2*padding - kernel_size + 1
        #   0 = 2*padding - kernel_size + 1
        #   -2*padding = -kernel_size + 1
        #   2*padding = kernel_size - 1
        #   padding = (kernel_size - 1) / 2
        #
        # For odd kernel sizes, this gives:
        #   3x3: padding = (3-1)/2 = 1
        #   5x5: padding = (5-1)/2 = 2
        #   7x7: padding = (7-1)/2 = 3
        #
        # Verification with 3x3 kernel, padding=1, input=32:
        #   output = floor((32 + 2*1 - 3) / 1) + 1 = floor(31) + 1 = 32 ✓
        padding = kernel_size // 2

        # Main convolution path
        self.conv1 = ConvAct(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                             use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum)
        self.conv2 = ManualConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, use_builtin=use_builtin_conv)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        # Shortcut connection (identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv to match dimensions when needed
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum)
            )

        # Final activation after addition
        if use_prelu:
            self.final_act = nn.PReLU(out_channels if prelu_channel_wise else 1)
        else:
            self.final_act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        # Add shortcut
        out += self.shortcut(identity)
        out = self.final_act(out)

        return out

class BottleneckBlock(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152.

    Uses 1x1 -> 3x3 -> 1x1 convolutions to reduce parameters while increasing depth.
    The first 1x1 reduces channels by 4x, 3x3 processes at lower dimension,
    then final 1x1 expands back to the output dimension.

    This is more efficient than basic blocks for deep networks:
    - Basic block (256->256): 2 * (3*3*256*256) = 1.18M params
    - Bottleneck (256->256): 1*1*256*64 + 3*3*64*64 + 1*1*64*256 = 70K params

    Supports stochastic depth (DropPath) for regularization:
    - Randomly drops the entire residual branch during training
    - Output becomes just the shortcut (identity or projection)
    - Drop probability increases linearly with depth (linear decay rule)
    - Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """

    # Expansion factor: output channels = planes * expansion
    # This is ALWAYS 4 for standard ResNet bottleneck blocks.
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, use_prelu=False, use_builtin_conv=False, prelu_channel_wise=True, bn_momentum=0.1, drop_path=0.0):
        """
        Args:
            in_channels: Number of input channels
            planes: Number of bottleneck (reduced) channels.
                   Called "planes" because each channel is a 2D (H×W) feature map.
                   The output will be planes * 4 channels.
            stride: Stride for the 3x3 convolution
            use_prelu: Whether to use PReLU instead of ReLU
            use_builtin_conv: Whether to use PyTorch's built-in convolutions
            prelu_channel_wise: Whether to use channel-wise PReLU parameters
            bn_momentum: Momentum for batch normalization
            drop_path: Drop path rate for stochastic depth (0.0 = no dropping)
        """
        super().__init__()

        # Output channels = planes * 4 (the expansion factor)
        out_channels = planes * self.expansion

        # Bottleneck architecture: 1x1 reduce -> 3x3 process -> 1x1 expand
        # Conv1: REDUCE channels from in_channels to planes
        self.conv1 = ManualConv2d(in_channels, planes, kernel_size=1, stride=1, padding=0, use_builtin=use_builtin_conv)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)

        # Conv2: PROCESS at reduced dimension (planes to planes)
        # Note: stride is applied here for spatial downsampling
        self.conv2 = ManualConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, use_builtin=use_builtin_conv)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)

        # Conv3: EXPAND from planes to out_channels (planes * 4)
        self.conv3 = ManualConv2d(planes, out_channels, kernel_size=1, stride=1, padding=0, use_builtin=use_builtin_conv)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        # Activation functions
        if use_prelu:
            self.act1 = nn.PReLU(planes if prelu_channel_wise else 1)
            self.act2 = nn.PReLU(planes if prelu_channel_wise else 1)
            self.final_act = nn.PReLU(out_channels if prelu_channel_wise else 1)
        else:
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
            self.final_act = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 conv to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum)
            )

        # ReZero: Learnable residual scaling parameter
        # Initialized to 0 to start with identity mapping, then learns optimal scaling
        # This prevents magnitude accumulation and enables training of very deep networks
        # Reference: "ReZero is All You Need" (Bachlechner et al., 2020)
        self.residual_scale = nn.Parameter(torch.zeros(1))

        # Stochastic Depth (DropPath)
        # Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
        #
        # WHAT IT DOES:
        # During training, randomly drops the entire residual branch with probability p.
        # When dropped: output = shortcut (the residual contribution is zeroed out)
        # When kept: output = shortcut + residual (normal ResNet behavior)
        #
        # HOW DROPPATH WORKS:
        # 1. Sample a random mask per sample in the batch (Bernoulli with p=drop_path)
        # 2. If mask=0: zero out the residual path entirely for that sample
        # 3. If mask=1: scale residual by 1/(1-p) to maintain expected value
        #    This scaling ensures E[output] is the same during training and inference
        #
        # WHY IT HELPS:
        # 1. Regularization: Like dropout, but at block level instead of neuron level
        # 2. Implicit ensemble: Training samples see networks of varying depths
        #    - Some samples train a 50-layer network, others a 30-layer, etc.
        #    - At inference, we get an ensemble average of all these depths
        # 3. Gradient flow: Shorter paths (when blocks dropped) have better gradients
        #    - Reduces vanishing gradient in very deep networks
        # 4. Reduces co-adaptation: Blocks can't rely on specific earlier blocks
        #
        # LINEAR DECAY RULE (from original paper):
        # - First block: drop_path = 0 (never dropped, always trained)
        # - Last block: drop_path = max_rate (e.g., 0.1 = 10% drop chance)
        # - Intermediate: linearly interpolated based on depth
        # Rationale: Earlier layers learn fundamental features (edges, textures)
        # that all images need. Later layers learn high-level combinations that
        # may be more redundant and benefit from regularization.
        #
        # AT INFERENCE:
        # DropPath is disabled (all blocks active), but the learned weights
        # already account for the stochastic training via the 1/(1-p) scaling.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        # Bottleneck path
        out = self.conv1(x)  # Reduce
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)  # Process
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)  # Expand
        out = self.bn3(out)

        # ReZero: Scale residual by learned parameter (starts at 0, learns optimal scaling)
        # softplus ensures scaling is always positive (prevents destructive interference)
        out = F.softplus(self.residual_scale) * out

        # Stochastic depth: randomly drop the residual path during training
        # When dropped, this block becomes an identity mapping (output = shortcut)
        out = self.drop_path(out)

        # Add shortcut (identity or projection) to (possibly dropped) residual
        out = self.shortcut(identity) + out
        out = self.final_act(out)

        return out

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

    def __init__(self, use_prelu: bool = False, *, use_builtin_conv: bool = True, prelu_channel_wise: bool = True, num_classes: int = 1000, bn_momentum: float = 0.1, drop_path_rate: float = 0.0, gradient_checkpointing: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.bn_momentum = bn_momentum
        self.drop_path_rate = drop_path_rate
        self.gradient_checkpointing = gradient_checkpointing

        # Warn if using slow manual convolutions
        if not use_builtin_conv:
            warnings.warn(
                "CNN is using slow manual convolution implementations. "
                "Set use_builtin_conv=True for faster training with PyTorch's built-in convolutions.",
                UserWarning,
                stacklevel=2
            )

        # Stochastic depth: calculate per-block drop rates using linear decay rule
        # ResNet-50 has 16 bottleneck blocks total: [3, 4, 6, 3] = 16 blocks
        # Linear decay: drop_rate increases from 0 at block 0 to drop_path_rate at block 15
        # Formula: drop_rate[i] = drop_path_rate * i / (total_blocks - 1)
        #
        # Example with drop_path_rate=0.1 and 16 blocks:
        #   Block 0:  0.0    (first block never dropped)
        #   Block 7:  0.047  (middle blocks have ~5% drop rate)
        #   Block 15: 0.1    (last block has full 10% drop rate)
        total_blocks = 3 + 4 + 6 + 3  # 16 blocks for ResNet-50
        drop_rates = [drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0  # Counter to track which block we're creating
        
        # Spatial dimension flow through the network (ResNet-50):
        # 1. Input: 224×224
        # 2. conv1 (7×7, stride=2, pad=3): (224 + 2*3 - 7)/2 + 1 = 112×112
        # 3. MaxPool (3×3, stride=2, pad=1): (112 + 2*1 - 3)/2 + 1 = 56×56
        # 4. conv2 all 3 blocks (stride=1): stays 56×56
        # 5. conv3 first BottleneckBlock (stride=2): (56 - 1)/2 + 1 = 28×28
        # 6. conv3 remaining 3 blocks (stride=1): stay 28×28
        # 7. conv4 first BottleneckBlock (stride=2): (28 - 1)/2 + 1 = 14×14
        # 8. conv4 remaining 5 blocks (stride=1): stay 14×14
        # 9. conv5 first BottleneckBlock (stride=2): (14 - 1)/2 + 1 = 7×7
        # 10. conv5 remaining 2 blocks (stride=1): stay 7×7

        # Layer 1: 7×7 conv, 64 filters, stride=2 (ImageNet input: 224×224)
        # We keep max pooling here (but replace it with stride=2 convs in later layers) because:
        # - Early layers detect simple features (edges/textures) where max pooling works well
        # - Rapid spatial reduction (112→56) reduces computation for all subsequent layers
        # - Standard practice in ResNet, VGG, and the original PReLU paper
        # - Only deeper layers with complex features benefit from learned downsampling
        self.conv1 = nn.Sequential(
            ConvAct(3, 64, kernel_size=7, stride=2, padding=3, use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum),  # 224×224 → 112×112
            ManualMaxPool2d(kernel_size=3, stride=2, padding=1, use_builtin=use_builtin_conv)  # 112×112 → 56×56
        )

        # Improved architecture based on ResNet principles:
        # - Replace 2×2 kernels → 3×3 (better receptive field growth)
        # - Add residual connections (solves vanishing gradients)
        # - Use stride=2 convolutions instead of max pooling (learnable downsampling)
        #   Reference: "Striving for Simplicity: The All Convolutional Net" (2014)
        #   Shows stride=2 convs preserve more information than max pooling

        # ResNet-50 architecture: [3, 4, 6, 3] blocks per stage with bottlenecks
        # Total layers: 1 (conv1) + 3×3 + 4×3 + 6×3 + 3×3 = 1 + 9 + 12 + 18 + 9 = 49 conv layers
        # (Plus 1 final FC = 50 total)

        # conv2_x: 3 bottleneck blocks
        # First block: 64 → 64 planes (256 output channels), stride=1 (no downsampling yet)
        # Remaining blocks: 256 → 64 planes (256 output channels)
        conv2_blocks = []
        conv2_blocks.append(BottleneckBlock(64, 64, stride=1,
                          use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
        block_idx += 1
        for _ in range(2):
            conv2_blocks.append(BottleneckBlock(256, 64, stride=1,
                            use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
            block_idx += 1
        self.conv2 = nn.Sequential(*conv2_blocks)  # Stays 56×56

        # conv3_x: 4 bottleneck blocks
        # First block: 256 → 128 planes (512 output channels), stride=2 for downsampling
        # Remaining blocks: 512 → 128 planes (512 output channels)
        conv3_blocks = []
        conv3_blocks.append(BottleneckBlock(256, 128, stride=2,  # 56×56 → 28×28
                          use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
        block_idx += 1
        for _ in range(3):
            conv3_blocks.append(BottleneckBlock(512, 128, stride=1,
                            use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
            block_idx += 1
        self.conv3 = nn.Sequential(*conv3_blocks)  # Stays 28×28

        # conv4_x: 6 bottleneck blocks
        # First block: 512 → 256 planes (1024 output channels), stride=2 for downsampling
        # Remaining blocks: 1024 → 256 planes (1024 output channels)
        conv4_blocks = []
        conv4_blocks.append(BottleneckBlock(512, 256, stride=2,  # 28×28 → 14×14
                          use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
        block_idx += 1
        for _ in range(5):
            conv4_blocks.append(BottleneckBlock(1024, 256, stride=1,
                            use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
            block_idx += 1
        self.conv4 = nn.Sequential(*conv4_blocks)  # Stays 14×14

        # conv5_x: 3 bottleneck blocks
        # First block: 1024 → 512 planes (2048 output channels), stride=2 for downsampling
        # Remaining blocks: 2048 → 512 planes (2048 output channels)
        conv5_blocks = []
        conv5_blocks.append(BottleneckBlock(1024, 512, stride=2,  # 14×14 → 7×7
                          use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
        block_idx += 1
        for _ in range(2):
            conv5_blocks.append(BottleneckBlock(2048, 512, stride=1,
                            use_prelu=use_prelu, use_builtin_conv=use_builtin_conv, prelu_channel_wise=prelu_channel_wise, bn_momentum=bn_momentum, drop_path=drop_rates[block_idx]))
            block_idx += 1
        self.conv5 = nn.Sequential(*conv5_blocks)  # Stays 7×7

        # Global Average Pooling (standard ResNet-50)
        # Transforms: (batch, 2048, 7, 7) → (batch, 2048, 1, 1)
        #
        # How it works:
        # - For each of the 2048 channels independently:
        #   * Take the 7×7 spatial grid (49 values)
        #   * Compute the mean of those 49 values
        #   * Result: 1 scalar per channel
        # - Final output: 2048 features (one per channel)
        #
        # Why this regularizes:
        # - Enforces spatial invariance: model can't memorize object positions
        # - Zero learnable parameters (just averaging)
        # - Forces learning of location-independent semantic features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier: Simple linear layer (standard ResNet-50)
        # 2048 features → num_classes
        # Parameters: 2048 × 1000 = 2.05M (vs 511M with SPP!)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x=None, pixel_values=None, **kwargs):
        # Handle both positional and keyword arguments for compatibility with HuggingFace Trainer
        if x is not None:
            input_tensor = x
        elif pixel_values is not None:
            input_tensor = pixel_values
        else:
            # Try to get pixel_values from kwargs
            input_tensor = kwargs.get('pixel_values')
            if input_tensor is None:
                raise ValueError("No input tensor provided. Expected 'x' or 'pixel_values' argument.")

        # Clone input tensor to prevent CUDAGraphs overwrite issues with torch.compile()
        # This is necessary when using torch.compile() which uses CUDAGraphs for optimization
        # The clone ensures the tensor isn't overwritten between runs
        input_tensor = input_tensor.clone()

        x = self.conv1(input_tensor)

        # Gradient Checkpointing
        #
        # PROBLEM: During backprop, PyTorch needs activations from the forward pass
        # to compute gradients. Normally these are stored in memory:
        #
        #   Forward:  x → conv2 → a2 → conv3 → a3 → conv4 → a4 → conv5 → output
        #                        ↑          ↑          ↑
        #                   [stored]   [stored]   [stored]  ← Uses ~30GB for ResNet-50
        #
        # SOLUTION: Don't store activations. Recompute them during backward pass:
        #
        #   Forward:  x → conv2 → conv3 → conv4 → conv5 → output
        #                (activations discarded, only output kept)
        #
        #   Backward: When we need a3 to compute grad for conv4:
        #             1. Re-run forward from last checkpoint: conv3(a2) → a3
        #             2. Use a3 to compute gradient
        #             3. Discard a3 again
        #
        # TRADE-OFF:
        #   Memory: ~50-70% reduction (only store checkpoint boundaries, not all activations)
        #   Compute: ~20-30% slower (each segment's forward pass runs twice)
        #
        # WHY IT'S WORTH IT:
        #   - Memory savings allow 2x larger batch sizes
        #   - 2x batch = 2x fewer optimizer steps = ~40% faster overall
        #   - Net effect: faster training despite recomputation overhead
        #
        # use_reentrant=False: Newer, safer implementation that works with autograd
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.conv2, x, use_reentrant=False)
            x = checkpoint(self.conv3, x, use_reentrant=False)
            x = checkpoint(self.conv4, x, use_reentrant=False)
            x = checkpoint(self.conv5, x, use_reentrant=False)
        else:
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)

        # Apply global average pooling and flatten
        x = self.avgpool(x)  # (batch, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (batch, 2048)

        return self.fc(x)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with automatic handling of torch.compile() wrapper prefix.

        When a model is compiled with torch.compile(), it adds "_orig_mod." prefix to all
        parameter names. This method automatically detects and strips this prefix.

        Parameters
        ----------
        state_dict : dict
            State dictionary to load
        strict : bool, optional
            Whether to strictly enforce that the keys in state_dict match. Defaults to True.
        """
        # Handle torch.compile() wrapper prefix
        # When a model is compiled with torch.compile(), it adds "_orig_mod." prefix to all keys
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Call parent class load_state_dict
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_pretrained(cls, checkpoint_path, use_prelu=None, use_builtin_conv=True, prelu_channel_wise=True, num_classes=1000, device=None):
        """Load a trained CNN model from a checkpoint directory.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint directory containing 'model.safetensors'
        use_prelu : bool, optional
            Whether to use PReLU activation. If None, will try to infer from checkpoint path.
        use_builtin_conv : bool, optional
            Whether to use builtin convolutions. Defaults to True.
        prelu_channel_wise : bool, optional
            Whether to use channel-wise PReLU. Defaults to True.
        num_classes : int, optional
            Number of output classes. Defaults to 1000.
        device : str or torch.device, optional
            Device to load the model on. If None, will use CUDA if available.
            
        Returns
        -------
        CNN
            Loaded CNN model with trained weights
            
        Examples
        --------
        >>> # Load from specific checkpoint
        >>> model = CNN.from_pretrained("/path/to/checkpoint-900900")
        >>> 
        >>> # Load latest checkpoint from results directory
        >>> import os
        >>> results_dir = "/path/to/results/cnn_results_relu"
        >>> checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint-")]
        >>> latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        >>> model = CNN.from_pretrained(os.path.join(results_dir, latest))
        """
        import os
        from safetensors.torch import load_file
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(device, str):
            device = torch.device(device)
            
        print(f"🔄 Loading CNN model from checkpoint...")
        print(f"   Path: {checkpoint_path}")
        print(f"   Activation: {'PReLU' if use_prelu else 'ReLU'}")
        print(f"   Device: {device}")
        
        # Create model with specified architecture
        model = cls(
            use_prelu=use_prelu,
            use_builtin_conv=use_builtin_conv, 
            prelu_channel_wise=prelu_channel_wise,
            num_classes=num_classes
        )
        
        # Load the trained weights
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.safetensors not found in {checkpoint_path}")
            
        print(f"📥 Loading trained weights from: model.safetensors")
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
        
        # Move to device
        model = model.to(device)
        return model

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------------------------
# Data preprocessing functions
# -------------------------------------------------------
def preprocess_images(examples):
    """Preprocess images for ImageNet (optimized for tensor storage)."""
    # ImageNet normalization values (standard for pretrained models)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Try to access the images - this is where UTF-8 EXIF errors occur
    try:
        images = examples['image']  # This line can trigger UTF-8 EXIF errors
        labels_list = examples['label']
    except UnicodeDecodeError as utf8_error:
        print(f"UTF-8 EXIF error when accessing batch: {utf8_error}")
        print(f"Skipping entire batch due to EXIF decoding error")
        # Return empty results for this batch
        return {
            'pixel_values': torch.empty(0, 3, 224, 224),
            'labels': torch.empty(0, dtype=torch.long)
        }
    except Exception as e:
        print(f"Error accessing batch data: {e}")
        return {
            'pixel_values': torch.empty(0, 3, 224, 224),
            'labels': torch.empty(0, dtype=torch.long)
        }

    # Pre-allocate tensors for the batch
    batch_size = len(images)
    processed_images = torch.empty(batch_size, 3, 224, 224)
    processed_labels = torch.empty(batch_size, dtype=torch.long)

    for i, image in enumerate(images):  # Now we can safely iterate
        try:
            # Convert image to PIL RGB format
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            else:
                image = image.convert('RGB')
            
            # Resize to 224×224 (ImageNet standard)
            image = image.resize((224, 224), Image.BILINEAR)
            
            # Convert to tensor - at this point image is always RGB (3D HWC)
            image = torch.tensor(np.array(image), dtype=torch.float32)
            
            # Safety check: ensure tensor is 3D (HWC format)
            if image.dim() == 2:
                # If somehow we still have a 2D tensor, convert to RGB by repeating the channel
                image = image.unsqueeze(0).repeat(3, 1, 1)  # (H, W) -> (3, H, W)
            elif image.dim() == 3:
                # Rearrange dimensions from Height-Width-Channel (HWC) to Channel-Height-Width (CHW)
                # This is required because PyTorch expects images in CHW format, but PIL/OpenCV use HWC
                image = image.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected image tensor shape: {image.shape}")
            
            # Normalize pixel values from [0, 255] range to [0, 1] range
            # Neural networks work better with normalized inputs, and 255.0 ensures float division
            image = image / 255.0
            
            # Apply ImageNet normalization using tensor operations
            image = (image - mean.view(3, 1, 1)) / std.view(3, 1, 1)
                
            # Store in pre-allocated tensor
            processed_images[i] = image
            processed_labels[i] = labels_list[i]
            
        except Exception as e:
            import traceback
            
            # Skip this image if any error occurs (including UTF-8 encoding errors)
            print(f"Warning: Skipping image {i} due to error: {type(e).__name__}: {e}")
            
            # Enhanced debugging for UTF-8 errors
            if "utf-8" in str(e).lower() or "codec" in str(e).lower():
                print(f"UTF-8 Error Details:")
                print(f"  Image index: {i}")
                print(f"  Image type: {type(image)}")
                print(f"  Error: {e}")
                
                # Try to get more info about the problematic image
                try:
                    if hasattr(image, 'filename'):
                        print(f"  Image filename: {image.filename}")
                    elif isinstance(image, str):
                        print(f"  Image path: {image}")
                except:
                    pass
                
                # Print traceback for UTF-8 errors to see exactly where it fails
                print(f"  Full traceback:")
                tb_lines = traceback.format_exc().split('\n')
                for line in tb_lines:
                    if line.strip():
                        print(f"    {line}")

    return {
        'pixel_values': processed_images.numpy(),
        'labels': processed_labels.numpy()
    }

# -------------------------------------------------------
# Learning rate helpers
# -------------------------------------------------------

class CosineWithHardRestartsDecaySchedulerWithWarmup(_LRScheduler):
    """Cosine restarts with decaying peaks and optional momentum damping."""

    def __init__(
        self,
        optimizer,
        *,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 1.0,
        cycle_decay: float = 1.0,
        min_lr_ratio: float = 0.0,
        cycle_warmup_ratio: float = 0.0,
        damp_momentum_at_restart: bool = False,
    ) -> None:
        self.num_warmup_steps = max(0, int(num_warmup_steps))
        self.num_training_steps = max(1, int(num_training_steps))
        self.num_cycles = max(num_cycles, 1e-6)
        self.cycle_decay = max(cycle_decay, 0.0)
        self.min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)
        self.cycle_warmup_ratio = min(max(cycle_warmup_ratio, 0.0), 0.999999)
        self.damp_momentum_at_restart = damp_momentum_at_restart

        # Track which cosine cycle we are in so restarts can damp Adam buffers
        self._current_cycle_index = -1
        self._current_cycle_peak_ratio = 1.0

        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < 0:
            return list(self.base_lrs)

        multiplier = self._lr_multiplier(step)
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):  # type: ignore[override]
        super().step(epoch)
        self._maybe_damp_momentum()

    def _lr_multiplier(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return float(step) / max(1, self.num_warmup_steps)

        if step >= self.num_training_steps:
            return self.min_lr_ratio

        cycle_info = self._compute_cycle_info(step)
        if cycle_info is None:
            return self.min_lr_ratio

        cycle_index, within_cycle, cycle_peak_ratio, amplitude, effective_warmup = cycle_info

        if effective_warmup > 0.0 and within_cycle < effective_warmup:
            warm_progress = within_cycle / max(effective_warmup, 1e-9)
            return self.min_lr_ratio + amplitude * warm_progress

        if effective_warmup >= 0.999999:
            return self.min_lr_ratio

        cosine_progress = (within_cycle - effective_warmup) / max(1e-9, 1.0 - effective_warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return self.min_lr_ratio + amplitude * cosine

    def _compute_cycle_info(self, step: int):
        if step < self.num_warmup_steps or step >= self.num_training_steps:
            return None

        progress = (step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
        cycle_progress = progress * self.num_cycles
        cycle_index = math.floor(cycle_progress)
        within_cycle = cycle_progress - cycle_index
        cycle_peak_ratio = max(self.min_lr_ratio, self.cycle_decay ** cycle_index)
        amplitude = max(0.0, cycle_peak_ratio - self.min_lr_ratio)
        effective_warmup = 0.0 if cycle_index == 0 else self.cycle_warmup_ratio
        return cycle_index, within_cycle, cycle_peak_ratio, amplitude, effective_warmup

    def _maybe_damp_momentum(self) -> None:
        if not self.damp_momentum_at_restart:
            return

        cycle_info = self._compute_cycle_info(self.last_epoch)
        if cycle_info is None:
            return

        cycle_index, _, cycle_peak_ratio, _, _ = cycle_info
        if cycle_index == self._current_cycle_index:
            return

        prev_peak_ratio = self._current_cycle_peak_ratio
        self._current_cycle_index = cycle_index
        self._current_cycle_peak_ratio = cycle_peak_ratio

        if cycle_index == 0:
            return

        scale = cycle_peak_ratio / max(prev_peak_ratio, 1e-12)
        scale = min(scale, 1.0)
        if scale < 1.0:
            self._scale_momentum_buffers(scale)

    def _scale_momentum_buffers(self, scale: float) -> None:
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if not state:
                    continue
                if "exp_avg" in state:
                    # Shrink AdamW momentum to align with the new peak LR.
                    state["exp_avg"].mul_(scale)
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].mul_(scale)


def get_cosine_with_hard_restarts_decay_schedule_with_warmup(
    optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1.0,
    cycle_decay: float = 1.0,
    min_lr_ratio: float = 0.0,
    cycle_warmup_ratio: float = 0.0,
    damp_momentum_at_restart: bool = False,
):
    """Cosine-with-restarts schedule that decays the peak LR each cycle.

    Args mirror :func:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` while
    adding ``cycle_decay`` (shrink the restart peaks), ``min_lr_ratio`` (floor
    relative to the base LR), ``cycle_warmup_ratio`` (per-cycle warmup fraction),
    and ``damp_momentum_at_restart`` (scale AdamW momentum buffers when a cycle
    restarts to avoid a sharp loss spike).
    """

    return CosineWithHardRestartsDecaySchedulerWithWarmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        cycle_decay=cycle_decay,
        min_lr_ratio=min_lr_ratio,
        cycle_warmup_ratio=cycle_warmup_ratio,
        damp_momentum_at_restart=damp_momentum_at_restart,
    )


# -------------------------------------------------------
# Custom Trainer for CNN
# -------------------------------------------------------
class CNNTrainer(Trainer):
    """Custom trainer for CNN models with gradient anomaly logging."""

    def __init__(self, *args, error_log_path="gradient_anomalies.log",
                 logging_thresholds=None, train_sampler=None,
                 grad_histogram_config=None, mixup_debug_log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_log_path = error_log_path
        self.last_loss = None
        self.custom_train_sampler = train_sampler  # For Repeated Augmentation

        # MixUp/CutMix debug logging (optional)
        self.mixup_debug_log_path = mixup_debug_log_path
        self.mixup_debug_enabled = mixup_debug_log_path is not None
        if self.mixup_debug_enabled:
            import os
            os.makedirs(os.path.dirname(mixup_debug_log_path) if os.path.dirname(mixup_debug_log_path) else ".", exist_ok=True)
            with open(mixup_debug_log_path, 'w') as f:
                f.write("=" * 100 + "\n")
                f.write("MIXUP/CUTMIX DEBUG LOG\n")
                f.write("=" * 100 + "\n\n")

        # Gradient histogram monitoring (optional)
        self.grad_histogram = None
        self.grad_histogram_config = grad_histogram_config or {}
        self._grad_histogram_initialized = False

        # Logging thresholds - can be configured from training script
        default_thresholds = {
            # Gradient thresholds
            'grad_norm': 7.0,  # Log when grad norm exceeds this
            'param_norm': 100.0,  # Log individual param norms exceeding this

            # BatchNorm thresholds
            'bn_mean_abs': 10.0,  # running_mean absolute value
            'bn_var_mean': 100.0,  # running_var mean value (upper bound)
            'bn_var_max': 1000.0,  # running_var max value
            'bn_var_min': 0.01,  # running_var mean value (lower bound)

            # PReLU thresholds
            'prelu_alpha_max': 1.0,  # Maximum alpha value
            'prelu_alpha_min': -0.5,  # Minimum alpha value (negative threshold)
            'prelu_alpha_mean': 0.5,  # Mean alpha value
            'prelu_alpha_std': 0.5,  # Std of alpha values

            # Activation magnitude thresholds
            'act_abs_max': 50.0,  # Maximum absolute activation
            'act_abs_mean': 10.0,  # Mean absolute activation
            'act_std': 10.0,  # Activation standard deviation
            'act_growth': 1.2,  # Growth ratio between stages

            # Residual block thresholds
            'residual_main_to_shortcut_ratio': 1.5,  # Main path vs shortcut magnitude
            'residual_growth_from_addition': 2.0,  # Combined growth from addition
            'residual_combined_std': 10.0,  # Absolute combined std threshold
        }

        # Merge user-provided thresholds with defaults
        self.thresholds = default_thresholds.copy()
        if logging_thresholds:
            self.thresholds.update(logging_thresholds)

        # Keep grad_norm_threshold for backward compatibility
        self.grad_norm_threshold = self.thresholds['grad_norm']

        # For tracking activation magnitudes
        self.activation_stats = {}
        self.activation_hooks_installed = False

        # For tracking BatchNorm batch statistics (Hypothesis 2)
        self.bn_batch_stats = {}
        self.bn_hooks_installed = False

        # For tracking residual block statistics (Hypothesis 3)
        self.residual_stats = {}
        self.residual_hooks_installed = False

        # Initialize error log file (if enabled)
        self.gradient_logging_enabled = error_log_path is not None
        if self.gradient_logging_enabled:
            import os
            os.makedirs(os.path.dirname(error_log_path) if os.path.dirname(error_log_path) else ".", exist_ok=True)
            with open(error_log_path, 'w') as f:
                f.write("=" * 100 + "\n")
                f.write("GRADIENT ANOMALY LOG\n")
                f.write("=" * 100 + "\n\n")

    def _get_train_sampler(self, dataset=None):
        """Override to use custom sampler (e.g., RepeatAugSampler for repeated augmentation)."""
        if self.custom_train_sampler is not None:
            return self.custom_train_sampler
        return super()._get_train_sampler(dataset)

    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use default collation for evaluation (no MixUp/CutMix)."""
        from torch.utils.data import default_collate

        # Temporarily replace the data_collator with default_collate
        original_collator = self.data_collator
        self.data_collator = default_collate

        # Call parent method which will use default_collate
        eval_dataloader = super().get_eval_dataloader(eval_dataset)

        # Restore the original collator for training
        self.data_collator = original_collator

        return eval_dataloader

    def _log_anomaly(self, step, message, details=None):
        """Log anomaly to error log file."""
        if not self.gradient_logging_enabled:
            return

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.error_log_path, 'a') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"[{timestamp}] Step {step}: {message}\n")
            f.write(f"{'='*100}\n")
            if details:
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        if not isinstance(pixel_values, torch.Tensor):
            if isinstance(pixel_values, list):
                # Check if list contains tensors (optimized format) or numpy arrays (legacy format)
                if len(pixel_values) > 0 and isinstance(pixel_values[0], torch.Tensor):
                    # Optimized format: list of tensors -> stack directly
                    pixel_values = torch.stack(pixel_values)
                else:
                    # Legacy format: list of numpy arrays -> convert to tensors first
                    pixel_values = torch.stack([torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in pixel_values])
            else:
                # Convert numpy array to tensor (legacy format)
                pixel_values = torch.from_numpy(pixel_values)

        # Convert labels to tensor if it's not already
        # Note: MixUp/CutMix produces soft labels (float, shape [N, C])
        # while standard training uses hard labels (long, shape [N])
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        # Move tensors to same device as model
        model_device = next(model.parameters()).device
        pixel_values = pixel_values.to(model_device)
        labels = labels.to(model_device)

        # Detect soft labels from MixUp/CutMix (2D float tensor)
        # vs hard labels from standard training (1D integer tensor)
        is_soft_labels = labels.dim() == 2

        # Pre-compute soft-label stats so we can both log and sanity check them
        label_sums = None
        lambda_vals = None
        if is_soft_labels:
            label_sums = labels.sum(dim=-1)
            lambda_vals = labels.max(dim=-1).values
            # Raise an anomaly early if the soft labels are not a valid distribution
            if torch.any((label_sums < 0.99) | (label_sums > 1.01)):
                self._log_anomaly(
                    self.state.global_step,
                    "Soft labels do not sum to 1.0 (MixUp/CutMix)",
                    {
                        "min_sum": f"{label_sums.min().item():.4f}",
                        "max_sum": f"{label_sums.max().item():.4f}",
                        "mean_sum": f"{label_sums.mean().item():.4f}",
                    },
                )

        log_mixup_stats = (
            self.mixup_debug_enabled
            and hasattr(self, "state")
            and (self.state.global_step < 10 or self.state.global_step % 500 == 0)
        )

        # MixUp/CutMix debug logging
        if log_mixup_stats:
            mode = "training" if model.training else "evaluation"
            with open(self.mixup_debug_log_path, 'a') as f:
                f.write(f"\n[Step {self.state.global_step}] {mode} - Labels info:\n")
                f.write(f"  Shape: {labels.shape}, Dtype: {labels.dtype}, Is soft: {is_soft_labels}\n")
                if is_soft_labels:
                    f.write(f"  Label sums (should be ~1.0): min={label_sums.min():.4f}, max={label_sums.max():.4f}, mean={label_sums.mean():.4f}\n")
                    f.write(f"  Label maxs (lambda values): min={lambda_vals.min():.4f}, max={lambda_vals.max():.4f}, mean={lambda_vals.mean():.4f}\n")
                else:
                    f.write(f"  Hard labels range: min={labels.min()}, max={labels.max()}\n")

        # Emit TensorBoard scalars so we can spot MixUp/CutMix pathologies in the UI
        if is_soft_labels and hasattr(self, "state") and (self.state.global_step < 10 or self.state.global_step % 200 == 0):
            self.log(
                {
                    "mixup/label_sum_min": label_sums.min().item(),
                    "mixup/label_sum_max": label_sums.max().item(),
                    "mixup/label_sum_mean": label_sums.mean().item(),
                    "mixup/lambda_min": lambda_vals.min().item(),
                    "mixup/lambda_max": lambda_vals.max().item(),
                    "mixup/lambda_mean": lambda_vals.mean().item(),
                }
            )

        # Check for input anomalies (both training and eval)
        mode = "training" if model.training else "evaluation"
        if torch.isnan(pixel_values).any():
            self._log_anomaly(
                self.state.global_step,
                f"NaN detected in input pixel_values ({mode})",
                {"batch_shape": str(pixel_values.shape)}
            )
        if torch.isinf(pixel_values).any():
            self._log_anomaly(
                self.state.global_step,
                f"Inf detected in input pixel_values ({mode})",
                {"batch_shape": str(pixel_values.shape)}
            )

        outputs = model(pixel_values)

        # Check for output anomalies (both training and eval)
        if torch.isnan(outputs).any():
            self._log_anomaly(
                self.state.global_step,
                f"NaN detected in model outputs ({mode})",
                {"output_shape": str(outputs.shape)}
            )
        if torch.isinf(outputs).any():
            self._log_anomaly(
                self.state.global_step,
                f"Inf detected in model outputs ({mode})",
                {"output_shape": str(outputs.shape)}
            )

        # Unified loss computation using PyTorch's CrossEntropyLoss
        # Works for BOTH hard labels [N] and soft labels [N, C] (PyTorch 1.10+)
        #
        # Hard labels (MixUp disabled):
        #   labels shape: [N] integer class indices
        #   CrossEntropyLoss converts to one-hot internally
        #   Apply label_smoothing from training args
        #
        # Soft labels (MixUp/CutMix enabled):
        #   labels shape: [N, C] soft probabilities from mixing
        #   CrossEntropyLoss computes cross-entropy against soft targets
        #   Do NOT apply label_smoothing (soft labels already encode mixing)

        if is_soft_labels:
            # Verify soft labels sum to 1.0 (critical for correct loss)
            if self.mixup_debug_enabled and not torch.allclose(label_sums, torch.ones_like(label_sums), atol=1e-3):
                with open(self.mixup_debug_log_path, 'a') as f:
                    f.write(f"⚠️ WARNING [Step {self.state.global_step}]: Soft labels don't sum to 1.0! min={label_sums.min():.4f}, max={label_sums.max():.4f}, mean={label_sums.mean():.4f}\n")

        # Get label smoothing factor (only applied to hard labels)
        label_smoothing = 0.0
        if not is_soft_labels and hasattr(self, 'args') and hasattr(self.args, 'label_smoothing_factor'):
            label_smoothing = self.args.label_smoothing_factor

        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = loss_fn(outputs, labels)

        # MixUp/CutMix loss comparison logging
        if log_mixup_stats:
            mode = "training" if model.training else "evaluation"
            with open(self.mixup_debug_log_path, 'a') as f:
                f.write(f"[Step {self.state.global_step}] {mode} - Loss: {loss.item():.4f} (is_soft_labels={is_soft_labels})\n")
                if is_soft_labels:
                    # Get label smoothing factor for fair comparison
                    ls_factor = getattr(self.args, 'label_smoothing_factor', 0.0)
                    hard_labels = labels.argmax(dim=-1)
                    hard_loss_no_smooth = nn.CrossEntropyLoss()(outputs.detach(), hard_labels)
                    hard_loss_with_smooth = nn.CrossEntropyLoss(label_smoothing=ls_factor)(outputs.detach(), hard_labels)

                    # Loss breakdown by primary vs secondary class
                    with torch.no_grad():
                        probs = F.softmax(outputs.detach(), dim=-1)
                        primary_idx = labels.argmax(dim=-1)
                        primary_probs = probs.gather(1, primary_idx.unsqueeze(1)).squeeze()

                        # Secondary class (second highest weight in soft labels)
                        labels_no_primary = labels.clone()
                        labels_no_primary.scatter_(1, primary_idx.unsqueeze(1), 0)
                        secondary_idx = labels_no_primary.argmax(dim=-1)
                        secondary_probs = probs.gather(1, secondary_idx.unsqueeze(1)).squeeze()
                        secondary_weights = labels_no_primary.max(dim=-1).values

                    f.write(f"  Soft label loss: {loss.item():.4f}\n")
                    f.write(f"  Hard label loss (no smooth): {hard_loss_no_smooth.item():.4f}, (smooth={ls_factor}): {hard_loss_with_smooth.item():.4f}\n")
                    f.write(f"  Diff from hard (no smooth): {loss.item() - hard_loss_no_smooth.item():.4f}, (smooth): {loss.item() - hard_loss_with_smooth.item():.4f}\n")
                    f.write(f"  Lambda (primary weight): min={lambda_vals.min():.4f}, max={lambda_vals.max():.4f}, mean={lambda_vals.mean():.4f}\n")
                    f.write(f"  Secondary weight (1-lambda): min={secondary_weights.min():.4f}, max={secondary_weights.max():.4f}, mean={secondary_weights.mean():.4f}\n")
                    f.write(f"  Model P(primary): min={primary_probs.min():.4f}, max={primary_probs.max():.4f}, mean={primary_probs.mean():.4f}\n")
                    f.write(f"  Model P(secondary): min={secondary_probs.min():.4f}, max={secondary_probs.max():.4f}, mean={secondary_probs.mean():.4f}\n")

        # Check for loss anomalies (both training and eval)
        if torch.isnan(loss):
            self._log_anomaly(
                self.state.global_step,
                f"NaN loss detected ({mode})",
                {"loss_value": "NaN"}
            )
        elif torch.isinf(loss):
            self._log_anomaly(
                self.state.global_step,
                f"Inf loss detected ({mode})",
                {"loss_value": "Inf"}
            )
        elif loss.item() > 100.0:
            self._log_anomaly(
                self.state.global_step,
                f"Abnormally large loss detected ({mode})",
                {"loss_value": f"{loss.item():.4f}"}
            )

        # Check for sudden loss spikes (training only since eval doesn't update sequentially)
        if model.training:
            if self.last_loss is not None and loss.item() > self.last_loss * 10:
                self._log_anomaly(
                    self.state.global_step,
                    "Loss spike detected (>10x increase)",
                    {
                        "previous_loss": f"{self.last_loss:.4f}",
                        "current_loss": f"{loss.item():.4f}",
                        "ratio": f"{loss.item() / self.last_loss:.2f}x"
                    }
                )
            self.last_loss = loss.item()
        
        # NOTE: Disabled - this was incorrectly weakening gradients by 1/grad_accum_steps
        # HuggingFace Trainer already handles gradient accumulation correctly
        # if model.training and hasattr(self, 'args') and hasattr(self.args, 'gradient_accumulation_steps'):
        #     if self.args.gradient_accumulation_steps > 1:
        #         loss = loss / self.args.gradient_accumulation_steps

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None):
        """Inject a cosine-restart scheduler with decay when requested."""

        optimizer = optimizer or self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer must be set before creating a scheduler.")

        args = self.args
        lr_kwargs = getattr(args, "lr_scheduler_kwargs", None) or {}
        use_decay_schedule = (
            args.lr_scheduler_type == "cosine_with_restarts"
            and (
                "cycle_decay" in lr_kwargs
                or "min_lr_ratio" in lr_kwargs
                or "cycle_warmup_ratio" in lr_kwargs
                or "damp_momentum_at_restart" in lr_kwargs
            )
        )

        if use_decay_schedule:
            if (
                self.lr_scheduler is None
                or getattr(self, "lr_scheduler_num_training_steps", None) != num_training_steps
            ):
                warmup_steps = args.get_warmup_steps(num_training_steps)
                self.lr_scheduler = get_cosine_with_hard_restarts_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                    num_cycles=lr_kwargs.get("num_cycles", 1.0),
                    cycle_decay=lr_kwargs.get("cycle_decay", 1.0),
                    min_lr_ratio=lr_kwargs.get("min_lr_ratio", 0.0),
                    cycle_warmup_ratio=lr_kwargs.get("cycle_warmup_ratio", 0.0),
                    damp_momentum_at_restart=lr_kwargs.get("damp_momentum_at_restart", False),
                )
                self.lr_scheduler_num_training_steps = num_training_steps
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)

    def _init_grad_histogram(self, model):
        """Initialize gradient histogram monitoring if configured."""
        if self._grad_histogram_initialized or not self.grad_histogram_config.get('enabled', False):
            return

        from grad_monitor import GradHistogram

        config = self.grad_histogram_config

        self.grad_histogram = GradHistogram(
            model,
            modules_filter=config.get('modules_filter', None),  # None = all layers
            bins=config.get('bins', 41),
            track_layers=config.get('track_layers', ("weight",)),
            log_every_n_steps=config.get('log_every_n_steps', 100),
        )
        self._grad_histogram_initialized = True

    def _export_grad_histogram(self):
        """Export gradient histogram to CSV (append mode)."""
        if self.grad_histogram is None:
            return

        config = self.grad_histogram_config
        output_dir = config.get('output_dir', self.args.output_dir)
        path = f"{output_dir}/grad_histogram.csv"
        self.grad_histogram.export_csv(path, append=True)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to add gradient monitoring."""
        # Initialize gradient histogram on first step (if enabled)
        if not self._grad_histogram_initialized and self.grad_histogram_config.get('enabled', False):
            self._init_grad_histogram(model)

        # Install hooks on first training step (only if gradient logging is enabled)
        # These hooks add overhead on every forward pass, so skip them when not logging
        if self.gradient_logging_enabled:
            if not self.activation_hooks_installed:
                self._install_activation_hooks(model)
            if not self.bn_hooks_installed:
                self._install_batchnorm_hooks(model)
            if not self.residual_hooks_installed:
                self._install_residual_hooks(model)

        # Set current step for gradient histogram (before backward pass)
        if self.grad_histogram is not None:
            self.grad_histogram.set_step(self.state.global_step)

        # Mark step beginning for CUDAGraphs compatibility with torch.compile()
        # This prevents CUDAGraphs from overwriting tensor outputs between runs
        if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Clamp PReLU alpha values to [0, 1] after optimizer step
        # This prevents pathological values like negative alphas (sign flipping)
        # or alphas > 1 (asymmetric amplification)
        self._clamp_prelu_alphas(model)

        # After backward pass, check gradients and batch norm stats
        if model.training and self.state.global_step % self.args.logging_steps == 0:
            self._check_gradients(model)
            self._check_batchnorm_stats(model, mode="training")
            self._check_prelu_params(model)
            self._check_activation_magnitudes()
            self._check_residual_blocks()

        # Export gradient histograms periodically (if enabled)
        if self.grad_histogram is not None:
            export_every = self.grad_histogram_config.get('export_every_n_steps', 1000)
            if self.state.global_step > 0 and self.state.global_step % export_every == 0:
                self._export_grad_histogram()

        return loss

    def train(self, *args, **kwargs):
        """Override train to cleanup gradient histogram at the end."""
        try:
            result = super().train(*args, **kwargs)
        finally:
            # Export remaining records and cleanup
            if self.grad_histogram is not None:
                self._export_grad_histogram()
                self.grad_histogram.close()
        return result

    def _clamp_prelu_alphas(self, model):
        """Clamp PReLU alpha parameters to [0, 1] range to prevent pathological behavior.

        - Alpha < 0: Flips sign of negative inputs (breaks monotonicity)
        - Alpha > 1: Amplifies negatives more than positives (asymmetric growth)
        - Valid range [0, 1]: 0=ReLU-like, 0.25=typical PReLU, 1=linear
        """
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.PReLU):
                    module.weight.clamp_(min=0.0, max=1.0)

    def _install_activation_hooks(self, model):
        """Install forward hooks to monitor activation magnitudes at each conv stage."""
        if self.activation_hooks_installed:
            return

        import re

        def make_hook(stage_name):
            def hook(module, input, output):
                # Capture output statistics
                self.activation_stats[stage_name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'abs_mean': output.abs().mean().item(),
                    'abs_max': output.abs().max().item(),
                }
            return hook

        # Find and hook all conv stages (conv1, conv2, conv3, etc.)
        conv_stages = []
        for name, module in model.named_modules():
            # Match module names like 'conv1', 'conv2', 'conv3', etc.
            if re.match(r'^conv\d+$', name):
                conv_stages.append(name)
                module.register_forward_hook(make_hook(f'{name}_output'))

        # Sort to ensure consistent ordering
        conv_stages.sort()
        self.conv_stage_names = conv_stages
        self.activation_hooks_installed = True

    def _install_batchnorm_hooks(self, model):
        """Install hooks to capture batch statistics from BatchNorm layers during training."""
        if self.bn_hooks_installed:
            return

        def make_bn_hook(layer_name):
            def hook(module, input, output):
                if module.training:
                    # Compute batch statistics from input
                    x = input[0]
                    # Compute mean and var across batch and spatial dimensions
                    # For BatchNorm2d: x shape is (N, C, H, W)
                    # We compute stats over dims [0, 2, 3] to get per-channel stats
                    batch_mean = x.mean(dim=[0, 2, 3])
                    batch_var = x.var(dim=[0, 2, 3], unbiased=False)

                    self.bn_batch_stats[layer_name] = {
                        'batch_mean': batch_mean.detach(),
                        'batch_var': batch_var.detach(),
                    }
            return hook

        # Hook all BatchNorm2d layers
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.register_forward_hook(make_bn_hook(name))

        self.bn_hooks_installed = True

    def _install_residual_hooks(self, model):
        """Install hooks to monitor residual block paths (Hypothesis 3: residual accumulation)."""
        if self.residual_hooks_installed:
            return

        # Find all BottleneckBlock modules
        for name, module in model.named_modules():
            if isinstance(module, BottleneckBlock):
                # Initialize storage for this block
                block_stats = {
                    'input': None,
                    'main_path': None,
                    'shortcut': None,
                    'combined': None,
                }
                self.residual_stats[name] = block_stats

                # Hook 1: Capture input to the block
                def make_input_hook(block_name):
                    def hook(module, input, output):
                        if module.training:
                            x = input[0]
                            self.residual_stats[block_name]['input'] = {
                                'std': x.std().item(),
                                'mean': x.mean().item(),
                                'abs_max': x.abs().max().item(),
                            }
                    return hook

                module.register_forward_hook(make_input_hook(name))

                # Hook 2: Capture main path output (after bn3, before addition)
                def make_main_path_hook(block_name):
                    def hook(module, input, output):
                        if module.training:
                            self.residual_stats[block_name]['main_path'] = {
                                'std': output.std().item(),
                                'mean': output.mean().item(),
                                'abs_max': output.abs().max().item(),
                            }
                    return hook

                module.bn3.register_forward_hook(make_main_path_hook(name))

                # Hook 3: Capture shortcut output
                if len(module.shortcut) > 0:  # Only if shortcut is not identity
                    def make_shortcut_hook(block_name):
                        def hook(module, input, output):
                            if module.training:
                                self.residual_stats[block_name]['shortcut'] = {
                                    'std': output.std().item(),
                                    'mean': output.mean().item(),
                                    'abs_max': output.abs().max().item(),
                                }
                        return hook

                    module.shortcut.register_forward_hook(make_shortcut_hook(name))
                else:
                    # Identity shortcut - copy input stats
                    def make_identity_shortcut_hook(block_name):
                        def hook(module, input, output):
                            if module.training and self.residual_stats[block_name]['input']:
                                self.residual_stats[block_name]['shortcut'] = self.residual_stats[block_name]['input'].copy()
                        return hook

                    module.register_forward_hook(make_identity_shortcut_hook(name))

                # Hook 4: Capture combined output (before final activation)
                # Use pre-hook on final_act to get value right after addition
                def make_combined_hook(block_name):
                    def hook(module, input):
                        if module.training:
                            combined = input[0]
                            self.residual_stats[block_name]['combined'] = {
                                'std': combined.std().item(),
                                'mean': combined.mean().item(),
                                'abs_max': combined.abs().max().item(),
                            }
                    return hook

                module.final_act.register_forward_pre_hook(make_combined_hook(name))

        self.residual_hooks_installed = True

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to check batch norm stats after evaluation."""
        # Run normal evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Check batch norm stats after evaluation completes (once per epoch)
        self._check_batchnorm_stats(self.model, mode="evaluation")

        return output

    def _check_gradients(self, model):
        """Check for gradient anomalies and log them."""
        step = self.state.global_step
        total_norm = 0.0
        nan_params = []
        inf_params = []
        large_grad_params = []
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check for NaN gradients
                if torch.isnan(param.grad).any():
                    nan_params.append(name)

                # Check for Inf gradients
                if torch.isinf(param.grad).any():
                    inf_params.append(name)

                # Compute per-parameter gradient norm
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                # Track per-layer norms (group by layer name prefix)
                layer_name = '.'.join(name.split('.')[:2])  # e.g., "conv1.conv"
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = 0.0
                layer_norms[layer_name] += param_norm ** 2

                # Check for abnormally large gradients in individual parameters
                if param_norm > self.thresholds['param_norm']:
                    large_grad_params.append((name, param_norm))

        total_norm = total_norm ** 0.5

        # Compute per-layer norms
        for layer_name in layer_norms:
            layer_norms[layer_name] = layer_norms[layer_name] ** 0.5

        # Log anomalies
        if nan_params:
            self._log_anomaly(
                step,
                f"NaN gradients detected in {len(nan_params)} parameters",
                {"parameters": ", ".join(nan_params[:10])}  # Log first 10
            )

        if inf_params:
            self._log_anomaly(
                step,
                f"Inf gradients detected in {len(inf_params)} parameters",
                {"parameters": ", ".join(inf_params[:10])}
            )

        if total_norm > self.grad_norm_threshold:
            # Find layers with largest gradients
            top_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:5]

            details = {
                "total_grad_norm": f"{total_norm:.4f}",
                "threshold": f"{self.grad_norm_threshold:.4f}",
                "learning_rate": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            }

            # Add top layers
            for i, (layer_name, norm) in enumerate(top_layers, 1):
                details[f"top_{i}_layer"] = f"{layer_name} (norm: {norm:.4f})"

            self._log_anomaly(
                step,
                f"Large gradient norm detected: {total_norm:.4f}",
                details
            )

        if large_grad_params:
            top_params = sorted(large_grad_params, key=lambda x: x[1], reverse=True)[:5]
            details = {}
            for i, (param_name, norm) in enumerate(top_params, 1):
                details[f"param_{i}"] = f"{param_name} (norm: {norm:.4f})"

            self._log_anomaly(
                step,
                f"Large gradients in {len(large_grad_params)} individual parameters",
                details
            )

    def _check_batchnorm_stats(self, model, mode="training"):
        """Check BatchNorm layers for statistical anomalies.

        Args:
            model: The model to check
            mode: "training" or "evaluation" for logging context
        """
        step = self.state.global_step
        bad_bn_layers = []

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                issues = []

                # Check running_mean
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    if torch.isnan(module.running_mean).any():
                        issues.append("running_mean has NaN")
                    elif torch.isinf(module.running_mean).any():
                        issues.append("running_mean has Inf")
                    else:
                        mean_abs = module.running_mean.abs().mean().item()
                        if mean_abs > self.thresholds['bn_mean_abs']:
                            issues.append(f"running_mean abnormally large (avg abs: {mean_abs:.2f})")

                # Check running_var
                if hasattr(module, 'running_var') and module.running_var is not None:
                    if torch.isnan(module.running_var).any():
                        issues.append("running_var has NaN")
                    elif torch.isinf(module.running_var).any():
                        issues.append("running_var has Inf")
                    elif (module.running_var <= 0).any():
                        issues.append("running_var has non-positive values")
                    else:
                        var_mean = module.running_var.mean().item()
                        var_max = module.running_var.max().item()
                        if var_mean > self.thresholds['bn_var_mean']:
                            issues.append(f"running_var abnormally large (avg: {var_mean:.2f})")
                        if var_max > self.thresholds['bn_var_max']:
                            issues.append(f"running_var max too large ({var_max:.2f})")
                        if var_mean < self.thresholds['bn_var_min']:
                            issues.append(f"running_var abnormally small (avg: {var_mean:.6f})")

                # Check weight and bias (gamma and beta parameters)
                if hasattr(module, 'weight') and module.weight is not None:
                    if torch.isnan(module.weight).any():
                        issues.append("weight has NaN")
                    elif torch.isinf(module.weight).any():
                        issues.append("weight has Inf")

                if hasattr(module, 'bias') and module.bias is not None:
                    if torch.isnan(module.bias).any():
                        issues.append("bias has NaN")
                    elif torch.isinf(module.bias).any():
                        issues.append("bias has Inf")

                if issues:
                    bad_bn_layers.append((name, issues))

        # Log all bad batch norm layers
        if bad_bn_layers:
            for layer_name, issues in bad_bn_layers:
                details = {
                    "layer": layer_name,
                    "mode": mode,
                    "issues": ", ".join(issues),
                    "num_issues": len(issues)
                }
                self._log_anomaly(
                    step,
                    f"BatchNorm anomaly detected in {layer_name} ({mode})",
                    details
                )

        # Check batch vs running variance mismatch (Hypothesis 2)
        if mode == "training" and self.bn_batch_stats:
            mismatch_layers = []

            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name in self.bn_batch_stats:
                    batch_stats = self.bn_batch_stats[name]

                    if module.running_var is not None:
                        batch_var_mean = batch_stats['batch_var'].mean().item()
                        running_var_mean = module.running_var.mean().item()

                        if running_var_mean > 0:
                            mismatch_ratio = abs(batch_var_mean - running_var_mean) / running_var_mean

                            # Only log if mismatch is significant (>30%)
                            if mismatch_ratio > 0.30:
                                mismatch_layers.append({
                                    'name': name,
                                    'batch_var_mean': batch_var_mean,
                                    'running_var_mean': running_var_mean,
                                    'mismatch_ratio': mismatch_ratio
                                })

            # Log batch vs running variance mismatches
            if mismatch_layers:
                for mismatch in mismatch_layers:
                    details = {
                        "layer": mismatch['name'],
                        "batch_var_mean": f"{mismatch['batch_var_mean']:.4f}",
                        "running_var_mean": f"{mismatch['running_var_mean']:.4f}",
                        "mismatch_ratio": f"{mismatch['mismatch_ratio']:.2%}",
                    }
                    self._log_anomaly(
                        step,
                        f"BatchNorm batch/running variance mismatch in {mismatch['name']}",
                        details
                    )

    def _check_prelu_params(self, model):
        """Check PReLU parameters for all layers."""
        step = self.state.global_step
        prelu_stats_all = []

        for name, module in model.named_modules():
            if isinstance(module, nn.PReLU):
                # Get PReLU weight (the learnable alpha parameter)
                if hasattr(module, 'weight') and module.weight is not None:
                    alpha = module.weight.data
                    issues = []

                    # Check for NaN/Inf
                    if torch.isnan(alpha).any():
                        issues.append("alpha has NaN")
                    elif torch.isinf(alpha).any():
                        issues.append("alpha has Inf")
                    else:
                        # Check statistics
                        alpha_mean = alpha.mean().item()
                        alpha_min = alpha.min().item()
                        alpha_max = alpha.max().item()
                        alpha_std = alpha.std().item()

                        # PReLU alpha should typically be small positive values (0.01-0.25)
                        # Large alphas mean strong amplification of negative activations
                        if alpha_max > self.thresholds['prelu_alpha_max']:
                            issues.append(f"alpha_max too large ({alpha_max:.4f})")
                        if alpha_min < self.thresholds['prelu_alpha_min']:
                            issues.append(f"alpha_min too negative ({alpha_min:.4f})")
                        if alpha_mean > self.thresholds['prelu_alpha_mean']:
                            issues.append(f"alpha_mean too large ({alpha_mean:.4f})")
                        if alpha_std > self.thresholds['prelu_alpha_std']:
                            issues.append(f"alpha_std too large ({alpha_std:.4f})")

                        # Store stats for all layers
                        prelu_stats_all.append({
                            'name': name,
                            'alpha_mean': alpha_mean,
                            'alpha_min': alpha_min,
                            'alpha_max': alpha_max,
                            'alpha_std': alpha_std,
                            'num_params': alpha.numel(),
                            'issues': issues
                        })

        # Log all PReLU stats (layers with issues first, then healthy layers)
        if prelu_stats_all:
            # Sort: problematic layers first, then by layer depth
            prelu_stats_all.sort(key=lambda x: (len(x['issues']) == 0, x['name']))

            for stats in prelu_stats_all:
                details = {
                    "layer": stats['name'],
                    "alpha_mean": f"{stats['alpha_mean']:.6f}",
                    "alpha_min": f"{stats['alpha_min']:.6f}",
                    "alpha_max": f"{stats['alpha_max']:.6f}",
                    "alpha_std": f"{stats['alpha_std']:.6f}",
                    "num_params": stats['num_params']
                }
                if stats['issues']:
                    details["issues"] = ", ".join(stats['issues'])
                    message = f"PReLU anomaly detected in {stats['name']}"
                else:
                    message = f"PReLU stats for {stats['name']}"

                self._log_anomaly(step, message, details)

    def _check_activation_magnitudes(self):
        """Check activation magnitudes at each conv stage to track magnitude accumulation."""
        step = self.state.global_step

        if not self.activation_stats or not hasattr(self, 'conv_stage_names'):
            return

        # Log activation statistics ONLY for stages with issues
        stage_outputs = [f'{stage}_output' for stage in self.conv_stage_names]

        for stage_output in stage_outputs:
            if stage_output in self.activation_stats:
                stats = self.activation_stats[stage_output]
                issues = []

                # Check for abnormally large activations
                if stats['abs_max'] > self.thresholds['act_abs_max']:
                    issues.append(f"abs_max too large ({stats['abs_max']:.2f})")
                if stats['abs_mean'] > self.thresholds['act_abs_mean']:
                    issues.append(f"abs_mean too large ({stats['abs_mean']:.2f})")
                if stats['std'] > self.thresholds['act_std']:
                    issues.append(f"std too large ({stats['std']:.2f})")

                # Only log if there are issues
                if issues:
                    details = {
                        "stage": stage_output,
                        "mean": f"{stats['mean']:.4f}",
                        "std": f"{stats['std']:.4f}",
                        "min": f"{stats['min']:.4f}",
                        "max": f"{stats['max']:.4f}",
                        "abs_mean": f"{stats['abs_mean']:.4f}",
                        "abs_max": f"{stats['abs_max']:.4f}",
                        "issues": ", ".join(issues)
                    }

                    message = f"Activation magnitude anomaly at {stage_output}"
                    self._log_anomaly(step, message, details)

        # Calculate growth rates and only log if any stage has excessive growth
        if len(stage_outputs) >= 2:
            growth_details = {}
            excessive_growth = False
            prev_std = None

            for stage_output in stage_outputs:
                if stage_output in self.activation_stats:
                    curr_std = self.activation_stats[stage_output]['std']

                    if prev_std is not None and prev_std > 0:
                        growth = curr_std / prev_std
                        growth_details[f"{stage_output}_growth"] = f"{growth:.4f}x"

                        # Flag if growth exceeds threshold
                        if growth > self.thresholds['act_growth']:
                            excessive_growth = True

                    prev_std = curr_std

            # Only log if there's excessive growth somewhere
            if excessive_growth and growth_details:
                self._log_anomaly(
                    step,
                    "Excessive activation magnitude growth detected",
                    growth_details
                )

    def _check_residual_blocks(self):
        """Check residual blocks for magnitude accumulation (Hypothesis 3)."""
        step = self.state.global_step

        if not self.residual_stats:
            return

        problematic_blocks = []

        for block_name, stats in self.residual_stats.items():
            # Skip if we don't have all the stats
            if not all([stats.get('input'), stats.get('main_path'), stats.get('shortcut'), stats.get('combined')]):
                continue

            input_std = stats['input']['std']
            main_std = stats['main_path']['std']
            shortcut_std = stats['shortcut']['std']
            combined_std = stats['combined']['std']

            issues = []

            # Check 1: Main path should be small corrections (main_std << shortcut_std)
            if shortcut_std > 0:
                main_to_shortcut_ratio = main_std / shortcut_std

                # If main path is dominating the shortcut
                threshold = self.thresholds['residual_main_to_shortcut_ratio']
                if main_to_shortcut_ratio > threshold:
                    issues.append(f"main path too large (ratio: {main_to_shortcut_ratio:.2f}, expected <{threshold})")

            # Check 2: Combined output should not grow excessively from addition
            if shortcut_std > 0:
                growth_from_addition = combined_std / shortcut_std

                # If combined grows excessively from addition, residuals are accumulating
                threshold = self.thresholds['residual_growth_from_addition']
                if growth_from_addition > threshold:
                    issues.append(f"excessive growth from addition ({growth_from_addition:.2f}x, expected <{threshold}x)")

            # Check 3: Absolute magnitude check
            if combined_std > self.thresholds['residual_combined_std']:
                issues.append(f"combined_std too large ({combined_std:.2f})")

            # Only log blocks with issues
            if issues:
                problematic_blocks.append({
                    'name': block_name,
                    'input_std': input_std,
                    'main_std': main_std,
                    'shortcut_std': shortcut_std,
                    'combined_std': combined_std,
                    'main_to_shortcut_ratio': main_std / shortcut_std if shortcut_std > 0 else 0,
                    'growth_from_addition': combined_std / shortcut_std if shortcut_std > 0 else 0,
                    'issues': issues
                })

        # Log problematic blocks
        if problematic_blocks:
            for block in problematic_blocks:
                details = {
                    "block": block['name'],
                    "input_std": f"{block['input_std']:.4f}",
                    "main_path_std": f"{block['main_std']:.4f}",
                    "shortcut_std": f"{block['shortcut_std']:.4f}",
                    "combined_std": f"{block['combined_std']:.4f}",
                    "main_to_shortcut_ratio": f"{block['main_to_shortcut_ratio']:.4f}",
                    "growth_from_addition": f"{block['growth_from_addition']:.4f}x",
                    "issues": ", ".join(block['issues'])
                }
                self._log_anomaly(
                    step,
                    f"Residual block anomaly detected in {block['name']}",
                    details
                )
