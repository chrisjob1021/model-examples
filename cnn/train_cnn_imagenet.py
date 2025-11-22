#!/usr/bin/env python3
"""Train ReLU CNN on ImageNet using ModelTrainer"""

import torch
from datasets import load_from_disk, load_dataset, Dataset
from transformers import TrainingArguments
import torchvision.transforms as T
import random
from torch.utils.data import default_collate
import argparse
from torch.utils.tensorboard import SummaryWriter

# Import from shared_utils package
from shared_utils import ModelTrainer, find_latest_checkpoint

from prelu_cnn import CNN, CNNTrainer
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.data.distributed_sampler import RepeatAugSampler

from PIL import Image, ImageFile
import warnings

# Handle corrupt images more gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Suppress EXIF warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Corrupt EXIF data")

class MixupCutmixCollator:
    """
    Custom collate function that applies MixUp and/or CutMix augmentation at the batch level
    using timm's battle-tested Mixup implementation.

    MixUp: Blends entire images and labels (img = λ*img1 + (1-λ)*img2)
    CutMix: Cuts rectangular patches from one image and pastes onto another

    When both are enabled, randomly switches between them based on switch_prob.

    Why these augmentations are effective:
    - MixUp: Smoother decision boundaries, better calibration, linear behavior between classes
    - CutMix: Forces model to learn from partial views, improves localization
    - Combined: Complementary regularization - MixUp for global mixing, CutMix for local mixing
    - Empirically improves ImageNet top-1 accuracy by 1-2%
    """

    def __init__(
        self,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=1000
    ):
        """
        Args:
            mixup_alpha: Beta distribution parameter for MixUp. Higher = more mixing.
                         Set to 0 to disable MixUp. Default 0.8 (timm standard).
            cutmix_alpha: Beta distribution parameter for CutMix patch sizes.
                          Set to 0 to disable CutMix. Default 1.0 (uniform patch sizes).
            prob: Probability of applying either augmentation to a batch.
                  Default 1.0 (always apply one of them).
            switch_prob: Probability of selecting CutMix over MixUp when both are active.
                         Default 0.5 (equal chance of each).
            mode: How to apply mixing:
                  - 'batch': Same mixing params for entire batch (fastest)
                  - 'pair': Different params per image pair
                  - 'elem': Different params per element (most diverse)
            label_smoothing: Label smoothing applied to mixed targets.
                             Set to 0.0 if using Trainer's label_smoothing_factor.
            num_classes: Number of classes for one-hot encoding.
        """
        self.mixup_enabled = mixup_alpha > 0 or cutmix_alpha > 0

        if self.mixup_enabled:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=prob,
                switch_prob=switch_prob,
                mode=mode,
                label_smoothing=label_smoothing,
                num_classes=num_classes
            )
        else:
            self.mixup_fn = None

        self.num_classes = num_classes

    def __call__(self, batch):
        """
        Collate function that creates a batch and applies MixUp/CutMix.

        Args:
            batch: List of examples from the dataset

        Returns:
            Dict with batched tensors, with MixUp/CutMix applied if enabled
        """
        # First, use default collate to create standard batch tensors
        batch_dict = default_collate(batch)

        # Apply MixUp/CutMix if enabled
        if self.mixup_fn is not None:
            # timm's Mixup expects (images, targets) and returns (mixed_images, mixed_targets)
            # mixed_targets are one-hot encoded with mixing applied
            mixed_images, mixed_labels = self.mixup_fn(
                batch_dict['pixel_values'],
                batch_dict['labels']
            )

            batch_dict['pixel_values'] = mixed_images
            batch_dict['labels'] = mixed_labels

        return batch_dict

class SafeImageNetDataset(Dataset):
    """
    Wrapper for ImageNet dataset that safely handles EXIF errors on-demand.
    Extends HuggingFace Dataset class to maintain compatibility.
    """
    def __init__(self, dataset, transform_fn=None):
        # Don't call super().__init__() to avoid Dataset initialization issues
        self.dataset = dataset
        self.transform_fn = transform_fn
        self.skipped_count = 0

        # Pre-cache known good images as fallbacks
        # Spread across dataset to get variety of classes
        self.fallback_indices = [0, 1000, 5000, 10000, 20000, 30000, 40000, 50000]
        self.fallback_cache = []
        self._initialize_fallback_cache()

        # Copy over essential internal attributes from the underlying dataset
        for attr in ['_data', '_info', '_split', '_indices', '_fingerprint']:
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))
    
    def _initialize_fallback_cache(self):
        """Pre-cache some known good images as fallbacks."""
        print("🔄 Initializing fallback cache for error handling...")
        for idx in self.fallback_indices:
            try:
                # Only cache up to available dataset size
                if idx >= len(self.dataset):
                    continue
                item = self.dataset[idx]
                if self.transform_fn:
                    item = self.transform_fn(item)
                self.fallback_cache.append(item)
            except Exception as e:
                # Skip bad fallback candidates
                continue

        if not self.fallback_cache:
            # Emergency: try first 100 indices to find at least one good image
            for idx in range(min(100, len(self.dataset))):
                try:
                    item = self.dataset[idx]
                    if self.transform_fn:
                        item = self.transform_fn(item)
                    self.fallback_cache.append(item)
                    break
                except:
                    continue

        print(f"✅ Cached {len(self.fallback_cache)} fallback images")

    def _get_fallback(self):
        """Return a real cached image as fallback."""
        if not self.fallback_cache:
            # This should never happen if initialization worked
            raise RuntimeError("No fallback images available - dataset may be corrupted")

        # Rotate through cached images to provide variety
        fallback_idx = self.skipped_count % len(self.fallback_cache)
        return self.fallback_cache[fallback_idx]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        # Handle column access (string key) - delegate to underlying dataset
        if isinstance(key, str):
            return self.dataset[key]

        # Handle row access (integer index) with error handling
        idx = key
        try:
            item = self.dataset[idx]

            # Apply transform if provided
            if self.transform_fn:
                item = self.transform_fn(item)

            return item

        except UnicodeDecodeError as e:
            self.skipped_count += 1
            print(f"⚠️ UnicodeDecodeError at index {idx}, using cached fallback (skipped total: {self.skipped_count})")
            return self._get_fallback()

        except Exception as e:
            self.skipped_count += 1
            print(f"⚠️ Error at index {idx}: {type(e).__name__}, using cached fallback (skipped total: {self.skipped_count})")
            return self._get_fallback()
    
    def __getitems__(self, indices):
        """Handle batch loading for efficient DataLoader operation."""
        batch = []
        for idx in indices:
            try:
                item = self.dataset[idx]

                # Apply transform if provided
                if self.transform_fn:
                    item = self.transform_fn(item)

                batch.append(item)
            except Exception as e:
                self.skipped_count += 1
                print(f"⚠️ Error at index {idx}: {type(e).__name__}, using cached fallback (skipped total: {self.skipped_count})")
                batch.append(self._get_fallback())
        return batch
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset."""
        return getattr(self.dataset, name)

def main():
    """Train ReLU CNN on ImageNet."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ReLU CNN on ImageNet")
    parser.add_argument("--no-logging", action="store_true", 
                        help="Disable timestamped logging folders and tensorboard reporting")
    args = parser.parse_args()
    
    # Set debugging options based on command line flags
    disable_logging = args.no_logging
    
    print("🚀 Training ReLU CNN on ImageNet")
    print("=" * 50)
    
    if disable_logging:
        print("📝 Logging disabled (no log folders will be created)")
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # L40S/Ada GPU-specific settings for numerical stability
    # TF32 is enabled by default on Ampere/Ada (A100, H100, L40S, RTX 30/40)
    # It uses reduced precision (10-bit mantissa vs 23-bit FP32) which can cause
    # numerical instability in BatchNorm and gradient computations
    if device.type == "cuda":
        # Print GPU architecture info for debugging
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

        # Disable TF32 (reduced precision tensor operations)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Force deterministic algorithms (critical for numerical stability on newer GPUs)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Prevent reduced precision reductions in FP32 mode
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        print(f"🔧 TF32 disabled for numerical stability")
        print(f"🔧 cuDNN deterministic mode enabled")
        print(f"🔧 Forced full FP32 precision for all operations")

    # Global flag to disable mixed precision training
    use_mixed_precision = False

    # Check for mixed precision support
    use_bf16 = False
    use_fp16 = False
    if use_mixed_precision and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Check for BF16 support (Ampere and newer: A100, H100, RTX 30/40 series)
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print(f"✅ BF16 (bfloat16) support detected - enabling mixed precision training")
        else:
            # Fallback to FP16 for older GPUs (Volta, Turing: V100, T4, RTX 20 series)
            use_fp16 = True
            print(f"✅ FP16 (float16) support detected - enabling mixed precision training")
            print(f"   (BF16 not available on this GPU, using FP16 fallback)")
    elif device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"⚠️ Mixed precision training disabled (use_mixed_precision=False)")
    
    if False:
        # Load training dataset from processed version
        dataset_path = "./processed_datasets/imagenet_processor"
        dataset = load_from_disk(dataset_path)
        
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    else:
        # Load from Hugging Face cache with specific version to avoid re-download
        train_dataset = load_dataset("imagenet-1k", split="train", revision="1.0.0")
        eval_dataset = load_dataset("imagenet-1k", split="validation", revision="1.0.0")

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    
    # Data augmentation parameters (DeiT-B recipe)
    random_erasing_prob = 0.25  # DeiT-B uses 0.25

    # Repeated Augmentation (DeiT-B uses this)
    # Each image is repeated N times in the epoch, each with different augmentation.
    # This increases augmentation diversity without loading more unique images.
    # Standard value is 3 repeats.
    num_aug_repeats = 3

    # RandAugment "9/0.5" from DeiT-B:
    #   9 = magnitude (intensity of transforms, scale 0-10)
    #   0.5 = mstd (magnitude std - adds Gaussian noise to magnitude per-transform)
    # Using timm's implementation to get mstd support
    randaug_config = 'rand-m9-mstd0.5'
    randaug_hparams = {'img_mean': tuple([int(x * 255) for x in mean])}
    randaug_transform = rand_augment_transform(randaug_config, randaug_hparams)

    # MixUp/CutMix parameters (timm defaults for ImageNet)
    #
    # Both augmentations sample a mixing ratio λ from Beta(alpha, alpha) distribution:
    #   - MixUp:  blended_img = λ * img_a + (1-λ) * img_b
    #   - CutMix: paste (1-λ) area from img_b onto img_a
    #   - Labels: soft_label = λ * label_a + (1-λ) * label_b
    #
    # How alpha affects the Beta distribution of λ:
    #   alpha = 0.2: U-shaped, λ near 0 or 1 (mostly one image dominates)
    #   alpha = 0.8: Mild mixing, λ biased toward edges but smoother than 0.2
    #   alpha = 1.0: Uniform [0,1], any mixing ratio equally likely
    #   alpha = 2.0: Bell-shaped, λ clusters around 0.5 (always ~50/50 mix)
    #
    mixup_alpha = 0.8       # timm default for MixUp. Lower = less blending on average
    cutmix_alpha = 1.0      # timm default for CutMix. 1.0 = uniform patch sizes
    mix_prob = 1.0          # Probability of applying MixUp or CutMix to each batch
    mix_switch_prob = 0.5   # When both enabled: P(CutMix) vs P(MixUp). 0.5 = equal chance
    mix_mode = 'batch'      # 'batch': same λ for all samples (fast)
                            # 'pair': different λ per image pair
                            # 'elem': different λ per element (most diverse, slower)
    mix_label_smoothing = 0.0  # Disabled here; using Trainer's label_smoothing_factor instead

    # Logging thresholds for anomaly detection
    # These control when warnings are logged for model internals (gradients, activations, etc.)
    # Higher values = less sensitive (fewer warnings), lower values = more sensitive (more warnings)
    logging_thresholds = {
        # Gradient thresholds
        'grad_norm': 7.0,  # Log when total gradient norm exceeds this (typical: 3-7 for your model)
        'param_norm': 100.0,  # Log individual parameter norms exceeding this

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
        'act_growth': 1.2,  # Growth ratio between conv stages

        # Residual block thresholds (adjusted from 0.5/1.2 to 1.5/2.0 for stable training)
        'residual_main_to_shortcut_ratio': 1.5,  # Main path vs shortcut magnitude
        'residual_growth_from_addition': 2.0,  # Combined growth from residual addition
        'residual_combined_std': 10.0,  # Absolute combined std threshold
    }

    # Define the data augmentation and preprocessing pipeline for training images
    # Augmentations are applied in order - each builds on the previous transformations
    train_transform = T.Compose([
        # 1. RGB CONVERSION - Ensures consistent 3-channel input
        # Why: Some ImageNet images are grayscale (1 channel), but CNN expects RGB (3 channels)
        # Without this, training would crash on grayscale images
        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        
        # 2. RANDOM RESIZED CROP - Primary spatial augmentation
        # Why: Teaches model scale/translation invariance by showing objects at different sizes/positions
        # scale=(0.08, 1.0) means crop can be 8% to 100% of original image (aggressive cropping)
        # This is THE most important augmentation for ImageNet - forces model to recognize partial objects
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        
        # 3. RANDOM HORIZONTAL FLIP - Simple but effective augmentation (50% probability)
        # Why: Doubles effective dataset size, teaches left-right invariance
        # Critical for natural images where orientation doesn't matter (cat facing left = cat facing right)
        #T.RandomHorizontalFlip(), ## TODO: disabled
        
        # 4. RANDAUGMENT - AutoML-discovered augmentation policy (DeiT-B: 9/0.5)
        # Applies 2 random ops from a set of 14 (rotation, shearing, color shifts, etc.)
        # Using timm's implementation for mstd support (magnitude noise)
        randaug_transform,
        
        # 5. TENSOR CONVERSION - PIL Image → Tensor, scales [0,255] → [0,1]
        # Must happen before normalize and after PIL-based augmentations
        T.ToTensor(),
        
        # 6. NORMALIZATION - Centers data around 0 with unit variance
        # This is CRITICAL for deep network training. Here's why:
        #
        # WITHOUT NORMALIZATION:
        # - Raw pixel values are in [0, 255], already scaled to [0, 1] by ToTensor()
        # - But [0, 1] range causes problems:
        #   * Activations only use positive half of activation functions (e.g., tanh)
        #   * Gradient flow is biased - always positive inputs mean gradients have consistent sign
        #   * Deep networks compound this: each layer's output drifts further from zero
        #   * This causes "internal covariate shift" - each layer constantly adapts to changing input distributions
        #
        # THE MATH:
        # - After normalization: x_norm = (x - mean) / std
        # - For ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # - This transforms [0, 1] to approximately [-2, 2] with center at 0
        # - Example: A mid-gray pixel (0.5) becomes (0.5-0.485)/0.229 = 0.065 (near zero)
        #
        # WHY ZERO MEAN MATTERS:
        # 1. Activation functions work best around zero:
        #    - ReLU: Allows both positive and negative gradients
        #    - Tanh/Sigmoid: Maximum gradient at zero (derivative is highest)
        # 2. Gradient flow: Positive and negative values allow gradients to change sign
        # 3. Weight initialization assumes zero-centered inputs (e.g., Xavier/He initialization)
        #
        # WHY UNIT VARIANCE MATTERS (THE EXPRESSIVITY CONNECTION):
        # Consider a neuron with ReLU activation to see why variance controls expressivity:
        #
        # LOW VARIANCE INPUTS (e.g., all values in [0.45, 0.55] before normalization):
        # - After normalization & weights: z = wx + b clustered in tiny range
        # - Most neurons output nearly identical values
        # - Network loses discriminative power - different inputs produce similar outputs
        # - Information bottleneck: can't distinguish between classes
        # - Gradients become tiny → extremely slow learning
        #
        # HIGH VARIANCE INPUTS (e.g., values in [-100, 100]):
        # - After weights: z = wx + b has huge magnitude
        # - Many ReLUs completely dead (z < 0) or exploding (z >> 1)
        # - Dead ReLU problem: neurons permanently output 0, no gradient flow
        # - Exploding activations: gradients explode, training unstable
        # - Network becomes extremely sparse and brittle
        #
        # UNIT VARIANCE INPUTS (≈ [-2, 2] after normalization):
        # - After weights: balanced mix of positive and negative values
        # - ~50% of ReLUs active, ~50% zero (healthy sparsity)
        # - Gradients flow properly through active neurons
        # - Network maintains expressivity - different inputs produce different activation patterns
        # - Can learn complex decision boundaries
        #
        # Without proper normalization to unit variance:
        # - Network either compresses all inputs to similar outputs (low var)
        # - Or becomes too sparse/unstable to train (high var)
        # - Cannot learn the complex features needed to achieve low loss
        #
        # WHY IMAGENET STATISTICS (not just 0.5, 0.5, 0.5)?
        # - These are computed from millions of real images
        # - Natural images aren't perfectly gray-centered:
        #   * Red channel: mean=0.485 (slight red bias in natural scenes)
        #   * Green channel: mean=0.456 (most important for human vision)
        #   * Blue channel: mean=0.406 (sky/water pulls blue higher)
        # - Using dataset statistics ensures network sees properly centered data
        # - Pretrained models REQUIRE these exact values (they were trained with them)
        #
        # WHAT HAPPENS WITHOUT THIS:
        # - Slow convergence (10x more epochs needed)
        # - Higher chance of gradient explosion (need lower learning rates)
        # - Dead ReLU problem (neurons get stuck outputting zero)
        # - Poor transfer learning (can't use pretrained models)
        T.Normalize(mean, std),
        
        # 7. RANDOM ERASING (Cutout variant) - Masks random rectangles with random pixels
        # Why: Forces model to use context, prevents overfitting to specific features
        # p=0.1 (10% chance), scale=(0.02, 0.1) means 2-10% of image area erased
        # Applied AFTER normalization, so erased regions have random normalized values
        T.RandomErasing(p=random_erasing_prob), #scale=(0.02, 0.1)), # TODO: this is very conserative, usually T.RandomErasing(p=0.5, scale=(0.02, 0.33))
                                                                   # or more aggressive T.RandomErasing(p=0.6, scale=(0.02, 0.4))
    ])

    # Define the preprocessing pipeline for evaluation images (no heavy augmentation)
    eval_transform = T.Compose([
        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),  # Ensure 3 channels (convert grayscale to RGB)
        T.Resize(256),                                    # Resize shorter side to 256 pixels
        T.CenterCrop(224),                                # Crop the center 224x224 region
        T.ToTensor(),                                     # Convert to tensor and scale to [0, 1]
        T.Normalize(mean, std),                           # Normalize using ImageNet mean and std
    ])

    def train_transform_fn(examples):
        # Handle both single examples and batches
        if isinstance(examples['image'], list):
            # Batch processing
            examples["pixel_values"] = [train_transform(image) for image in examples["image"]]
        else:
            # Single example processing  
            examples["pixel_values"] = train_transform(examples["image"])

        # Convert 'label' to 'labels' to match HuggingFace Trainer expectations
        examples["labels"] = examples["label"]
        del examples["label"]
        
        # Remove the original image to avoid DataLoader issues
        del examples["image"]
        return examples

    def eval_transform_fn(examples):
        # Handle both single examples and batches
        if isinstance(examples['image'], list):
            # Batch processing
            examples["pixel_values"] = [eval_transform(image) for image in examples["image"]]
        else:
            # Single example processing
            examples["pixel_values"] = eval_transform(examples["image"])

        # There's some craziness going on within HuggingFace Trainer that requires this hack
        examples["labels"] = examples["label"]
        del examples["label"]

        # Remove the original image to avoid DataLoader issues
        del examples["image"]
        return examples

    # Wrap datasets with safe wrapper to handle EXIF errors on-demand
    train_dataset = SafeImageNetDataset(train_dataset, train_transform_fn)
    eval_dataset = eval_dataset.with_transform(eval_transform_fn)
    
    print(f"✅ Loaded datasets with safe EXIF error handling")
    print(f"✅ Training samples: {len(train_dataset):,}")
    print(f"✅ Validation samples: {len(eval_dataset):,}")
    
    use_prelu = True # TODO: Re-enable PReLU (set to True) to prevent dying ReLU in conv5 (conv5.0 output_std=0.0245 due to ReLU killing negative values)
                       # PReLU allows negative values to pass through (scaled by alpha), preventing dead neurons
                       # Combined with ReZero scaling, this should stabilize conv5 training

    # Stochastic depth (DropPath) rate
    # Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    # Drop rate increases linearly from 0 at first block to this value at last block
    # 0.1 = 10% drop probability at the deepest block (90% survival)
    drop_path_rate = 0.1

    # Create CNN model
    activation_type = "PReLU" if use_prelu else "ReLU"
    bn_momentum = 0.1  # Batch normalization momentum (lower = more stable running stats)
                        # TODO: adjusting this value between 0.01 and 0.1 (default) to avoid invalid batch norm stats
    print(f"\n🏗️ Creating {activation_type} CNN model ({1000} classes)...")
    print(f"🔧 Activation function: {activation_type}")
    print(f"🔧 BatchNorm momentum: {bn_momentum}")
    print(f"🔧 Stochastic depth (drop_path_rate): {drop_path_rate}")
    model = CNN(
        use_prelu=use_prelu,
        use_builtin_conv=True,  # Use fast PyTorch convolutions
        num_classes=1000,
        bn_momentum=bn_momentum,
        drop_path_rate=drop_path_rate
    )
    
    # This section moved below after we determine resume status
    
    # Force fresh weight initialization to ensure clean start
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        elif hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
    if False:
        model.apply(reset_weights)
        print("✅ Forced fresh weight initialization")
    
    # Move model to GPU
    model = model.to(device)
    print(f"✅ Model moved to device: {device}")
    
    # Verify model is on the correct CUDA device (cuda:0 if available)
    expected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_device = next(model.parameters()).device
    if model_device != expected_device:
        print(f"❌ Warning: Model parameters are on {model_device}, expected {expected_device}")
    else:
        print(f"✅ Model parameters confirmed on {expected_device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    batch_size_per_gpu = 256
    grad_accum = 4          
                                # Why not larger batches (e.g., 2048)?
                                # SHARP VS FLAT MINIMA:
                                # - Large batches: compute accurate gradients → go straight downhill → find nearest steep valley (sharp minimum)
                                # - Small batches: noisy gradients → bounce around → skip sharp valleys → settle in wide basins (flat minimum)
                                #
                                # Why sharp minima are bad:
                                # 1. Generalization: Test data slightly differs from train data. In sharp minimum, small perturbations cause huge loss spikes
                                # 2. Instability: Balanced on knife edge - any LR spike or gradient perturbation kicks you out → those loss spikes in TensorBoard
                                # 3. Train/eval gap: Model memorized training data's path to sharp valley, but test data falls off the cliff
                                #
                                # Batch 1024 provides good balance: stable gradients + enough noise to find flat, generalizable minima
    
    # Check if we want to resume from a checkpoint
    base_output_dir = f"./results/cnn_results_{'prelu' if use_prelu else 'relu'}"
    resume = False  # Set to True to resume from checkpoint, False for fresh training
    
    # Find checkpoint path if resuming
    if resume:
        checkpoint_path = find_latest_checkpoint(base_output_dir)
        if not checkpoint_path:
            print(f"⚠️ No checkpoint found in {base_output_dir}")
            print("Available options:")
            print("1. Set resume = False to start fresh training")
            print("2. Ensure checkpoint exists in the expected directory")
            raise ValueError(f"No checkpoint found for resume mode in {base_output_dir}")
    else:
        checkpoint_path = None
    
    # Configure training based on whether we're resuming
    if resume:
        # Extended training configuration when resuming
        num_epochs = 300
        output_dir = f"./results/cnn_resumed_{'prelu' if use_prelu else 'relu'}"
        
        # Load model weights ONLY (not optimizer/scheduler state)
        print(f"🔄 RESUME MODE: Found checkpoint at {checkpoint_path}")
        print(f"   Loading model weights ONLY (fresh optimizer/scheduler)")
        print(f"   Training for {num_epochs} additional epochs")
        print(f"   Output directory: {output_dir}")
        
        # Load the model weights manually
        import os
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        
        if os.path.exists(model_path):
            print(f"   Loading weights from: {model_path}")
            from safetensors.torch import load_file
            if model_path.endswith(".safetensors"):
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"   ✅ Model weights loaded successfully")
        else:
            print(f"   ⚠️ Model file not found at {checkpoint_path}")
            raise FileNotFoundError(f"Could not find model weights in {checkpoint_path}")
    else:
        # Original training configuration
        num_epochs = 300
        output_dir = base_output_dir
        print(f"🆕 Starting fresh training for {num_epochs} epochs")
        print(f"   Output directory: {output_dir}") 

    # For ImageNet training, I'd recommend sticking with batch_size=256 rather than 128. Here's why:

    #   1. Better gradient estimates: Larger batches provide more stable gradients
    #   2. Linear scaling rule: With batch=1024 and lr=0.1, you're at the sweet spot 
    #   3. Standard practice: Most successful ImageNet trainings use effective batch sizes of 256-2048

    #   If we switch to 128 for example:
    #   - Effective batch = 512 → more noisy gradients
    #   - Would need to reduce lr to ~0.05 for stability
    #   - Training would take longer (2x more gradient steps)
    #   - No accuracy benefit over batch=1024

    #   The linear scaling rule states that when you increase batch size,
    #   you can proportionally increase learning rate to maintain similar training dynamics.

    #   The formula: lr = base_lr × (batch_size / base_batch)
    #   Standard ImageNet baseline:
    #   - Base batch: 256
    #   - Base lr: 0.1

    #   Our setup:
    #   - Batch: 1024 (256×4 grad accum)
    #   - Expected lr: 0.1 × (1024/256) = 0.4

    #   But we're using lr=0.1, which is conservative and safer. Here's why this works:
    #   1. Linear scaling breaks down around batch 2k-8k depending on the model/dataset
    #   2. Beyond batch ~2k, you need additional tricks (warmup, LARS/LAMB optimizers) to maintain stability
    #   3. Our lr=0.1 with batch=1024 is actually the "sweet spot" because:
    #     - It's aggressive enough for fast convergence
    #     - Conservative enough to avoid instability
    #     - Matches many successful ImageNet papers' settings

    # Output directory is already set in the resume logic above

    # DeiT-B learning rate formula: 0.0005 × (batch_size / 512)
    # With batch_size=1024: lr = 0.0005 × 2 = 0.001
    effective_batch_size = batch_size_per_gpu * grad_accum
    initial_lr = 0.0005 * (effective_batch_size / 512)

    # DeiT-B warmup: 5 epochs
    warmup_epochs = 5
    warmup_ratio = warmup_epochs / num_epochs  # 5/300 ≈ 0.0167

    if resume:
        initial_lr = initial_lr * 0.3  # Reduce LR when resuming
        warmup_ratio = 0.01  # Shorter warmup for resume
        print(f"📈 Resumed training with reduced LR={initial_lr:.6f}")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,  # More epochs for better convergence
        per_device_train_batch_size=batch_size_per_gpu,  # Reduced for stability
        per_device_eval_batch_size=batch_size_per_gpu,
        learning_rate=initial_lr,
        weight_decay=0.05,  # DeiT-B uses 0.05
        warmup_ratio=warmup_ratio,  # Dynamic warmup based on resume status
        gradient_accumulation_steps=grad_accum,
        eval_steps=1,
        logging_steps=100,
        save_steps=1,
        seed=42,
        logging_dir="./logs/logs" if not disable_logging else None,
        remove_unused_columns=False, # Fix for custom dataset format
        dataloader_num_workers=16,      # Parallel data loading
        dataloader_persistent_workers=False,    # Enabled for L40S - keeps workers alive, reduces PCIe overhead on non-NVLink GPUs
                                                # IMPORTANT: it looks we were getting oom-killed leaving these alive on a 128GB mem system
        dataloader_pin_memory=True,     # If True, the DataLoader will copy Tensors into CUDA pinned memory before returning them.
                                        # This can speed up host-to-GPU transfer, especially for large batches.

        # Mixed precision training for 2-3x speedup and 50% memory reduction
        # BF16 (bfloat16): More stable, no loss scaling needed, available on Ampere+ GPUs (A100, H100, RTX 30/40)
        # FP16 (float16): Faster on older GPUs, requires loss scaling, available on Volta+ (V100, T4, RTX 20)
        bf16=use_bf16,
        fp16=use_fp16,

        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,  # Changing from 0.99 to 0.999 to be less sensitive to variations in gradients
        adam_epsilon=1e-08,
        # AdamW optimizer details:
        # - Decouples weight decay from gradient-based updates (better than Adam for vision)
        # - betas: (β1, β2) control exponential moving averages of gradients and squared gradients
        #   - β1=0.9: momentum for gradient (bias towards recent 10 steps)
        #   - β2=0.999: momentum for squared gradient (bias towards recent 1000 steps)
        # - eps=1e-08: small constant for numerical stability
        # - Learning rates for AdamW are typically 10-100x lower than SGD
        # === Momentum SGD variants (one parameter update per step) ====================
        # Notation:
        #   theta: params
        #   v:     velocity/momentum buffer (same shape as theta), init to 0
        #   mu:    momentum coefficient in [0, 1)  (e.g., 0.9–0.95)
        #   lr:    learning rate
        #   g:     gradient at current params, g = ∇f(theta)

        # 1) Heavy-ball (Polyak) momentum
        # v = mu * v - lr * g(theta)
        # theta += v
        #
        # Intuition: carry over a fraction (mu) of last step and subtract the current
        # gradient scaled by lr. Gradient is evaluated at the CURRENT point theta.

        # 2) Nesterov (look-ahead) momentum
        # theta_look = theta + mu * v        # peek ahead along momentum
        # g_star     = ∇f(theta_look)        # gradient at the look-ahead point
        # v          = mu * v - lr * g_star
        # theta     += v
        #
        # Intuition: same structure, but the gradient is taken at the anticipated
        # position.

        # 3) PyTorch implementation of Nesterov (single-pass form)
        # (what torch.optim.SGD(..., momentum=mu, nesterov=True, dampening=0) does)
        # b = mu * b + g(theta)               # momentum buffer (EMA of grads)
        # g_eff = g(theta) + mu * b           # effective Nesterov gradient
        # theta -= lr * g_eff                 # single update

        # --- Momentum buffer as an EMA of gradients -----------------------------------
        # Recurrence (PyTorch SGD with dampening=0):
        #   b_t = μ * b_{t-1} + g_t                    # g_t := ∇f(θ_t)
        #
        # Unrolled (shows the EMA weights explicitly):
        #   b_t = g_t + μ g_{t-1} + μ^2 g_{t-2} + ... + μ^t g_0
        #        = ∑_{k=0}^{t} μ^k * g_{t-k}
        #   # Geometric weights favor recent grads; effective memory ≈ 1/(1-μ) steps.
        #
        # Notes:
        # - Requires: momentum > 0 and dampening == 0 for true Nesterov behavior.
        # - torch.optim.SGD uses coupled L2 as weight_decay (adds λ*theta to g).

        # --- Momentum (Polyak heavy-ball) and the geometric-series factors ---
        # Update:
        #   v_{t+1} = μ v_t − η ∇f(θ_t)
        #   θ_{t+1} = θ_t + v_{t+1}
        #
        # Unrolled (geometric series on past grads):
        #   v_t = -η ∑_{k=0}^t μ^k g_{t-k}
        #   • Past gradients are weighted by μ^k (recent ones count more).

        # First-order decrease: f(θ + Δ) ≈ f(θ) + ∇f(θ)·Δ.
        # To reduce f with a small step, choose Δ that minimizes this linear term
        # under a size limit -> Δ = −η ∇f(θ). That is the steepest (Euclidean) descent.
        # Momentum just adds inertia: keep μ of last velocity (useful when directions persist, otherwise it resists),
        # then take the same downhill step −η ∇f(θ_t).

        max_grad_norm=None,  # DeiT-B disables gradient clipping
        lr_scheduler_type="cosine",  # DeiT-B uses cosine decay to 0
        # Alternative: cosine with minimum LR floor
        # lr_scheduler_type="cosine_with_min_lr",
        # lr_scheduler_kwargs={
        #     "min_lr_rate": 0.30,  # Minimum LR as ratio of initial LR (% of initial)
        #     # lr = min_lr + (initial_lr - min_lr) * (1 + cos(π * t/T)) / 2
        # },
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=False,  # Don't load previous checkpoints
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=False,
        label_names=["labels"], # need this to get eval_loss
        label_smoothing_factor=0.1,  # DeiT-B uses 0.1
        report_to="tensorboard" if not disable_logging else "none",
    )


    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")

    # Show mixed precision status
    if training_args.bf16:
        print(f"  Mixed precision: BF16 (bfloat16) ✅")
    elif training_args.fp16:
        print(f"  Mixed precision: FP16 (float16) ✅")
    else:
        print(f"  Mixed precision: Disabled (FP32)")

    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Weight decay: {training_args.weight_decay}")
    print(f"  WD/LR ratio: {training_args.weight_decay / training_args.learning_rate:.1f}")
    print(f"  Warmup ratio: {training_args.warmup_ratio}")
    print(f"  LR scheduler: {training_args.lr_scheduler_type}")
    if training_args.lr_scheduler_kwargs:
        print(f"  LR scheduler kwargs: {training_args.lr_scheduler_kwargs}")
    print(f"  Optimizer: {training_args.optim}")
    print(f"  Gradient clipping: {training_args.max_grad_norm}")
    print(f"  Label smoothing: {training_args.label_smoothing_factor}")
    print(f"  Evaluation strategy: {training_args.eval_strategy}")
    print(f"  Evaluation steps: {training_args.eval_steps}")
    print(f"  Save strategy: {training_args.save_strategy}")
    print(f"  Save steps: {training_args.save_steps}")
    print(f"  Save total limit: {training_args.save_total_limit}")
    print(f"  Logging strategy: {training_args.logging_strategy}")
    print(f"  Logging steps: {training_args.logging_steps}")
    print(f"  Output directory: {training_args.output_dir}")
    print(f"  Remove unused columns: {training_args.remove_unused_columns}")
    print(f"  Dataloader workers: {training_args.dataloader_num_workers}")
    print(f"  Load best model at end: {training_args.load_best_model_at_end}")
    print(f"  Metric for best model: {training_args.metric_for_best_model}")
    print(f"  Greater is better: {training_args.greater_is_better}")
    print(f"  Prediction loss only: {training_args.prediction_loss_only}")
    print(f"  Logging directory: {training_args.logging_dir}")
    print(f"  Report to: {training_args.report_to}")
    
    # Create MixUp/CutMix collator for training
    # Only used during training - evaluation uses default collation
    mixup_cutmix_collator = MixupCutmixCollator(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mix_prob,
        switch_prob=mix_switch_prob,
        mode=mix_mode,
        label_smoothing=mix_label_smoothing,
        num_classes=1000
    )
    
    # Create trainer using ModelTrainer
    print(f"\n🏋️ Setting up trainer...")

    # Set up error log path for gradient anomaly tracking
    import os
    error_log_path = os.path.join(output_dir, "gradient_anomalies.log")
    print(f"📝 Gradient anomaly log: {error_log_path}")

    # Create Repeated Augmentation sampler (DeiT-B)
    # This repeats each sample N times per epoch with different augmentations
    repeat_aug_sampler = RepeatAugSampler(train_dataset, num_repeats=num_aug_repeats)
    print(f"🔄 Repeated Augmentation: {num_aug_repeats}x repeats per image")

    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer_class=CNNTrainer,
        data_collator=mixup_cutmix_collator,  # MixUp/CutMix augmentation at batch level
        resume_from_checkpoint=None,  # Don't resume trainer state - we loaded weights manually
        trainer_kwargs={
            "error_log_path": error_log_path,  # Pass error log path for gradient anomaly tracking
            "logging_thresholds": logging_thresholds,  # Pass configurable logging thresholds
            "train_sampler": repeat_aug_sampler,  # Repeated augmentation sampler
        },
    )
    
    # Log hyperparameters to TensorBoard
    if not disable_logging and training_args.logging_dir:
        writer = SummaryWriter(log_dir=training_args.logging_dir)
        
        # Collect all hyperparameters
        hparams = {
            # Model architecture
            'model/activation': activation_type,
            'model/num_classes': 1000,
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
            'model/bn_momentum': bn_momentum,
            'model/drop_path_rate': drop_path_rate,

            # Training configuration
            'training/epochs': training_args.num_train_epochs,
            'training/batch_size_per_gpu': training_args.per_device_train_batch_size,
            'training/gradient_accumulation': training_args.gradient_accumulation_steps,
            'training/effective_batch_size': training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            'training/learning_rate': training_args.learning_rate,
            'training/weight_decay': training_args.weight_decay,
            'training/wd_lr_ratio': training_args.weight_decay / training_args.learning_rate,
            'training/warmup_ratio': training_args.warmup_ratio,
            'training/label_smoothing': training_args.label_smoothing_factor,
            'training/max_grad_norm': training_args.max_grad_norm,
            'training/seed': training_args.seed,
            'training/mixed_precision': 'bf16' if training_args.bf16 else ('fp16' if training_args.fp16 else 'fp32'),
            'training/use_bf16': 1 if training_args.bf16 else 0,
            'training/use_fp16': 1 if training_args.fp16 else 0,

            # Optimizer
            'optimizer/type': training_args.optim,
            'optimizer/adam_beta1': training_args.adam_beta1,
            'optimizer/adam_beta2': training_args.adam_beta2,
            'optimizer/adam_epsilon': training_args.adam_epsilon,

            # Scheduler
            'scheduler/type': training_args.lr_scheduler_type,

            # Data augmentation
            'augmentation/mixup_alpha': mixup_alpha,
            'augmentation/cutmix_alpha': cutmix_alpha,
            'augmentation/mix_prob': mix_prob,
            'augmentation/mix_switch_prob': mix_switch_prob,
            'augmentation/randaugment_ops': randaugment_ops,
            'augmentation/randaugment_magnitude': randaugment_magnitude,
            'augmentation/random_erasing_prob': random_erasing_prob,

            # Logging thresholds
            'logging/grad_norm_threshold': logging_thresholds['grad_norm'],
            'logging/param_norm_threshold': logging_thresholds['param_norm'],
            'logging/residual_main_to_shortcut_ratio': logging_thresholds['residual_main_to_shortcut_ratio'],
            'logging/residual_growth_from_addition': logging_thresholds['residual_growth_from_addition'],
            'logging/act_growth_threshold': logging_thresholds['act_growth'],

            # System
            'system/num_workers': training_args.dataloader_num_workers,
            'system/pin_memory': training_args.dataloader_pin_memory,
            'system/resume_mode': resume,
            'system/device': str(device),
        }
        
        # Log hyperparameters as scalars (only numeric values)
        for key, value in hparams.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f'hparams/{key}', value, 0)

        # Log all hyperparameters as text summary for easy viewing
        import json
        hparams_text = json.dumps(hparams, indent=2)
        writer.add_text('hyperparameters', f'```json\n{hparams_text}\n```', 0)

        writer.flush()
        writer.close()
        
        print(f"📊 Logged {len(hparams)} hyperparameters to TensorBoard")
        print(f"   View with: tensorboard --logdir {training_args.logging_dir}")
    
    # Run training
    trainer.run()
    
    print(f"💾 Model saved to: {training_args.output_dir}")
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()
