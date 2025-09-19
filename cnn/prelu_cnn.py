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
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional

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
                
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply SPP and flatten
        x = self.spp(x)  # (batch, 256, 50)
        
        return self.classifier(x)

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

def get_cosine_with_hard_restarts_decay_schedule_with_warmup(
    optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1.0,
    cycle_decay: float = 1.0,
    min_lr_ratio: float = 0.0,
    cycle_warmup_ratio: float = 0.0,
):
    """Cosine-with-restarts schedule that decays the peak LR each cycle.

    Hugging Face's built-in ``cosine_with_restarts`` scheduler resets the
    learning rate to its initial value at every restart. That works well when
    you only need a handful of restarts, but for long ImageNet runs it tends to
    blow up the loss because the peak LR never shrinks. This helper mirrors the
    original schedule while multiplying each cycle by ``cycle_decay`` and keeping
    a configurable ``min_lr_ratio`` floor relative to the initial LR.

    Args:
        optimizer: Wrapped optimizer whose LR should be scheduled.
        num_warmup_steps: Linear warmup steps before cosine kicks in.
        num_training_steps: Total training steps for the schedule horizon.
        num_cycles: Number of cosine cycles (can be fractional like 2.5).
        cycle_decay: Multiplicative decay applied after each restart.
        min_lr_ratio: Floor as a ratio of the initial LR (0 keeps the default).
        cycle_warmup_ratio: Fraction of each cosine cycle reserved for a linear
            warmup from ``min_lr_ratio`` up to the cycle peak (defaults to 0 for
            the standard cosine shape).
    """

    # Sanitize input parameters to prevent edge cases
    # Ensure at least a tiny positive number of cycles to avoid division by zero
    num_cycles = max(num_cycles, 1e-6)
    # Decay can't be negative (that would increase LR over time)
    cycle_decay = max(cycle_decay, 0.0)
    # min_lr_ratio must be between 0 and 1 (can't go below 0% or above 100% of initial LR)
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)
    # cycle_warmup_ratio must be less than 1 (can't warm up for entire cycle)
    # Using 0.999999 instead of 1.0 to avoid division by zero in cosine calculation
    cycle_warmup_ratio = min(max(cycle_warmup_ratio, 0.0), 0.999999)

    def lr_lambda(current_step: int) -> float:
        """Calculate learning rate multiplier for the given training step.
        
        This function returns a multiplier between 0 and 1 that gets applied
        to the optimizer's initial learning rate.
        """
        
        # PHASE 1: Initial warmup (before any cosine cycles)
        # Linear ramp from 0 to 1 over num_warmup_steps
        if current_step < num_warmup_steps:
            # Example: step 50 out of 100 warmup steps → returns 0.5 → LR = 0.5 * initial_lr
            return float(current_step) / max(1, num_warmup_steps)

        # PHASE 2: After all training is done
        # Maintain minimum learning rate
        if current_step >= num_training_steps:
            return min_lr_ratio

        # PHASE 3: Main cosine cycles with restarts
        
        # Calculate where we are in the overall schedule (0.0 to 1.0)
        # Example: step 1000 out of 10000 total steps (with 100 warmup) → progress = 0.1
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        
        # Scale progress by number of cycles to find our position across all cycles
        # Example: progress=0.4 with num_cycles=4 → cycle_progress = 1.6
        cycle_progress = progress * num_cycles
        
        # Determine which cycle we're in (0-indexed)
        # Example: cycle_progress=1.6 → cycle_index = 1 (we're in the 2nd cycle)
        cycle_index = math.floor(cycle_progress)
        
        # Find our position within the current cycle (0.0 to 1.0)
        # Example: cycle_progress=1.6 → within_cycle = 0.6 (60% through 2nd cycle)
        within_cycle = cycle_progress - cycle_index

        # Calculate target peak for this cycle after decay has been applied
        # Example: cycle_decay=0.55, cycle_index=1 → cycle_peak_ratio = 0.55
        cycle_peak_ratio = max(min_lr_ratio, cycle_decay ** cycle_index)

        # Calculate the amplitude (difference between peak and minimum)
        # Clamp to zero in case the decay pushes us below the minimum ratio
        # Example: min_lr_ratio=0.1, cycle_peak_ratio=0.55 → amplitude = 0.45
        amplitude = max(0.0, cycle_peak_ratio - min_lr_ratio)

        # SUB-PHASE 3A: Per-cycle warmup (if enabled)
        # Linear ramp at the start of each cycle
        # IMPORTANT: Do not apply per-cycle warmup to the very first cycle right after
        # the global warmup. The global warmup already brings LR to the cycle peak.
        # If we applied a per-cycle warmup here, LR would drop to min and ramp up again.
        effective_cycle_warmup_ratio = 0.0 if cycle_index == 0 else cycle_warmup_ratio
        if effective_cycle_warmup_ratio > 0.0 and within_cycle < effective_cycle_warmup_ratio:
            # Calculate progress through the warmup portion of this cycle
            # Example: within_cycle=0.05, cycle_warmup_ratio=0.1 → warm_progress=0.5
            warm_progress = within_cycle / max(effective_cycle_warmup_ratio, 1e-9)
            # Linear interpolation from min_lr_ratio to peak of current cycle
            # Example: min_lr_ratio=0.1, amplitude=0.45 → returns 0.1 + 0.45*0.5 = 0.325
            return min_lr_ratio + amplitude * warm_progress

        # Edge case: if entire cycle is warmup, just return minimum
        if effective_cycle_warmup_ratio >= 0.999999:
            return min_lr_ratio

        # SUB-PHASE 3B: Cosine annealing portion of the cycle
        # Calculate position in the cosine portion (after per-cycle warmup)
        # Example: within_cycle=0.6, cycle_warmup_ratio=0.1 → cosine_progress = 0.5/0.9 = 0.556
        cosine_progress = (within_cycle - effective_cycle_warmup_ratio) / max(1e-9, 1.0 - effective_cycle_warmup_ratio)
        
        # Apply cosine function (1 at start, 0 at end of cycle)
        # cos(0) = 1, cos(π) = -1, so 0.5*(1+cos(x)) maps to [1,0]
        # Example: cosine_progress=0.556 → cosine = 0.5*(1+cos(0.556π)) ≈ 0.206
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

        # Final LR multiplier: base minimum plus scaled cosine value
        # Example: min_lr_ratio=0.1, amplitude=0.45, cosine=0.206 → returns 0.1 + 0.45*0.206 ≈ 0.193
        return min_lr_ratio + amplitude * cosine

    # Return PyTorch LambdaLR scheduler that will call lr_lambda at each step
    return LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------
# Custom Trainer for CNN
# -------------------------------------------------------
class CNNTrainer(Trainer):
    """Custom trainer for CNN models."""

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
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        # Move tensors to same device as model
        model_device = next(model.parameters()).device
        pixel_values = pixel_values.to(model_device)
        labels = labels.to(model_device)

        outputs = model(pixel_values)
        
        # Use label smoothing from training args if available, otherwise default to 0.0
        label_smoothing = 0.0
        if hasattr(self, 'args') and hasattr(self.args, 'label_smoothing_factor'):
            label_smoothing = self.args.label_smoothing_factor
        
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = loss_fn(outputs, labels)
        
        # Fix for HuggingFace Trainer gradient accumulation scaling bug
        # Trainer scales logged loss by gradient_accumulation_steps, but actual training uses unscaled loss
        # Only apply this fix during training, not during evaluation
        if model.training and hasattr(self, 'args') and hasattr(self.args, 'gradient_accumulation_steps'):
            # Only scale for logging, not for actual training gradients
            if self.args.gradient_accumulation_steps > 1:
                # This ensures logged loss shows the correct per-sample loss
                loss = loss / self.args.gradient_accumulation_steps

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
                )
                self.lr_scheduler_num_training_steps = num_training_steps
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)
