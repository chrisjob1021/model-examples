#!/usr/bin/env python3
"""Train ReLU CNN on ImageNet using ModelTrainer"""

import torch
from datasets import load_from_disk, load_dataset, Dataset
from transformers import TrainingArguments
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import random
from torch.utils.data import default_collate
import argparse

# Import from shared_utils package
from shared_utils import ModelTrainer, find_latest_checkpoint

from prelu_cnn import CNN, CNNTrainer

class CutMixCollator:
    """
    Custom collate function that applies CutMix augmentation at the batch level.
    
    CutMix is a regularization strategy that:
    1. Cuts rectangular patches from one image
    2. Pastes them onto another image in the same batch
    3. Mixes labels proportionally based on the area of the patch
    
    Why CutMix is effective:
    - Forces model to learn from partial views (unlike MixUp which blends entire images)
    - Improves localization ability - model must identify objects from fragments
    - Provides stronger regularization than dropout or standard augmentation
    - Empirically improves ImageNet top-1 accuracy by 1-2%
    
    Applied at batch level because:
    - Requires pairs of images to mix
    - More efficient than image-level augmentation
    - Allows consistent mixing ratio across the batch
    """
    
    def __init__(self, num_classes=1000, alpha=1.0, prob=0.5):
        """
        Args:
            num_classes: Number of classes for one-hot encoding
            alpha: Beta distribution parameter controlling the size of cut patches.
                   
                   The mixing ratio λ is sampled from Beta(alpha, alpha):
                   - λ determines what fraction of the image to keep (1-λ is fraction replaced)
                   - Patch area = (1-λ) * image_area
                   
                   How alpha affects the distribution:
                   
                   alpha = 1.0 (default):
                   - Beta(1,1) = Uniform(0,1) distribution
                   - Equal probability for all patch sizes (0% to 100% of image)
                   - Most diverse training - can get tiny patches or nearly full replacement
                   - Example: λ could be 0.1 (90% replaced), 0.5 (50% replaced), or 0.9 (10% replaced) with equal probability
                   
                   alpha = 0.2 (more extreme):
                   - Beta(0.2, 0.2) is U-shaped - biased toward 0 or 1
                   - Tends to either replace almost nothing OR almost everything
                   - More aggressive augmentation - creates very easy or very hard samples
                   - Example: λ likely to be <0.1 (>90% replaced) or >0.9 (<10% replaced)
                   
                   alpha = 2.0 (more moderate):
                   - Beta(2,2) is bell-shaped - biased toward 0.5
                   - Tends to replace around 40-60% of the image
                   - More conservative - consistent difficulty level
                   - Example: λ clusters around 0.5 (roughly half the image replaced)
                   
                   alpha = 5.0 (very moderate):
                   - Beta(5,5) is strongly peaked at 0.5
                   - Almost always replaces 40-60% of the image
                   - Very predictable augmentation strength
                   - Less diversity in training samples
                   
                   Recommended values:
                   - alpha=1.0: Good default, maximum diversity
                   - alpha=0.5-0.8: If model struggles with training stability
                   - alpha=1.5-2.0: If model overfits and needs consistent regularization
                   
            prob: Probability of applying CutMix to a batch
        """
        self.cutmix = T2.CutMix(num_classes=num_classes, alpha=alpha)
        self.prob = prob
        self.num_classes = num_classes
        
    def __call__(self, batch):
        """
        Collate function that creates a batch and optionally applies CutMix.
        
        Args:
            batch: List of examples from the dataset
        
        Returns:
            Dict with batched tensors, potentially with CutMix applied
        """
        # First, use default collate to create standard batch tensors
        batch_dict = default_collate(batch)
        
        # Only apply CutMix during training with specified probability
        # Skip CutMix if labels are already mixed (from previous CutMix application)
        if random.random() < self.prob and len(batch_dict['labels'].shape) == 1:
            # CutMix returns:
            # - Mixed images with rectangular patches swapped
            # - One-hot encoded mixed labels (soft labels for mixed classes)
            mixed_images, mixed_labels = self.cutmix(
                batch_dict['pixel_values'], 
                batch_dict['labels']
            )
            
            batch_dict['pixel_values'] = mixed_images
            batch_dict['labels'] = mixed_labels
            
            # Note: HuggingFace Trainer will automatically handle one-hot labels
            # The loss function will compute cross-entropy with soft targets
        
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
        
        # Copy over essential internal attributes from the underlying dataset
        for attr in ['_data', '_info', '_split', '_indices', '_fingerprint']:
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))
    
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
            print(f"⚠️ UnicodeDecodeError error at index {idx}, trying next... (skipped: {self.skipped_count})")
            
        except Exception as e:
            self.skipped_count += 1
            print(f"⚠️ Error at index {idx}: {e}, trying next... (skipped: {self.skipped_count})")
    
        return {
            'pixel_values': torch.zeros(3, 224, 224),
            'label': 0
        }
    
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
                print(f"⚠️ Error at index {idx}: {e}, using fallback... (skipped: {self.skipped_count})")
                batch.append({
                    'pixel_values': torch.zeros(3, 224, 224),
                    'labels': 0
                })
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
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if False:
        # Load training dataset
        dataset_path = "./processed_datasets/imagenet_processor"
        dataset = load_from_disk(dataset_path)
        
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    else:
        train_dataset = load_dataset("imagenet-1k", split="train")
        eval_dataset = load_dataset("imagenet-1k", split="validation")

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

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
        T.RandomHorizontalFlip(),
        
        # 4. RANDAUGMENT - AutoML-discovered augmentation policy
        # Why: Applies 2 random ops from a set of 14 (rotation, shearing, color shifts, etc.)
        # magnitude=9 (out of 10) is aggressive - helps regularization but may slow initial convergence
        # This replaces manual tuning of individual augmentations
        T.RandAugment(num_ops=2, magnitude=7), # Initially near 9, this is near maximum (10 is max)
        
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
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
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
    
    use_prelu = True
    
    # Create CNN model
    activation_type = "PReLU" if use_prelu else "ReLU"
    print(f"\n🏗️ Creating {activation_type} CNN model ({1000} classes)...")
    print(f"🔧 Activation function: {activation_type}")
    model = CNN(
        use_prelu=use_prelu,
        use_builtin_conv=True,  # Use fast PyTorch convolutions
        num_classes=1000
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
        num_epochs = 600
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

    # Adjust learning rate for resume
    if resume:
        # When resuming, use a moderate LR since previous training plateaued
        # Previous training bottomed out at ~0.015, so we need higher than that
        initial_lr = 0.03  # 30% of original LR - aggressive enough to make progress
        warmup_ratio = 0.01  # Small 1% warmup for safety
        print(f"📈 Starting resumed training with LR={initial_lr}")
    else:
        initial_lr = 0.1
        warmup_ratio = 0.05  # Original 5% warmup for fresh training
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,  # More epochs for better convergence
        per_device_train_batch_size=batch_size_per_gpu,  # Reduced for stability
        per_device_eval_batch_size=batch_size_per_gpu,
        learning_rate=initial_lr,
        weight_decay=1e-4,
        # A reasonable weight_decay value typically ranges from 1e-4 to 1e-2
        #   Why these values work:
        #   - Too low (<1e-5): Minimal regularization effect, model may overfit
        #   - Too high (>1e-1): Over-regularization, model underfits and struggles to learn
        #   - Sweet spot (1e-4 to 1e-2): Balances learning capacity with generalization
        #   Weight decay (L2 regularization) penalizes large weights by adding λ||w||² to the loss, 
        #   encouraging the model to use smaller, more distributed weights rather than relying on a few large parameters. 

        # For vision models like CNNs, 1e-4 is often a good default. 
        # For transformers, values around 1e-2 are sometimes used, especially with AdamW optimizer 
        # which decouples weight decay from gradient-based updates.
        warmup_ratio=warmup_ratio,  # Dynamic warmup based on resume status
        gradient_accumulation_steps=grad_accum,
        eval_steps=1,
        logging_steps=100,
        save_steps=1,
        seed=42,
        logging_dir="./logs/logs" if not disable_logging else None,
        remove_unused_columns=False, # Fix for custom dataset format
        dataloader_num_workers=8, # Parallel data loading
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,     # If True, the DataLoader will copy Tensors into CUDA pinned memory before returning them.
                                        # This can speed up host-to-GPU transfer, especially for large batches.
        optim="sgd",
        optim_args="momentum=0.90,nesterov=True,dampening=0", # Tried 0.95, was trailing behind 0.90
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

        max_grad_norm=10.0, # grad norms are 4-6 during training, this just adds some protection
        lr_scheduler_type="cosine_with_min_lr",  # Use cosine with min LR for both fresh and resume
        lr_scheduler_kwargs={
            "min_lr": 0.15,  # 15% of initial LR as minimum
        },
        eval_strategy="steps",
        save_strategy="epoch",
        logging_strategy="steps",
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=False,  # Don't load previous checkpoints
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=False,
        label_names=["labels"], # need this to get eval_loss
        label_smoothing_factor=0.05,  # Label smoothing for regularization
        report_to="tensorboard" if not disable_logging else "none",
    )

    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Weight decay: {training_args.weight_decay}")
    print(f"  Warmup ratio: {training_args.warmup_ratio}")
    print(f"  LR scheduler: {training_args.lr_scheduler_type}")
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
    
    # Create CutMix collator for training
    # Only used during training - evaluation uses default collation
    cutmix_collator = CutMixCollator(
        num_classes=1000,  # ImageNet has 1000 classes
        alpha=1.0,         # Standard beta distribution for balanced mixing
        prob=0.5           # Apply CutMix to 50% of batches for balanced regularization
    )
    
    # Create trainer using ModelTrainer
    print(f"\n🏋️ Setting up trainer...")
    
    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer_class=CNNTrainer,
        #data_collator=cutmix_collator,  # Add CutMix collator for batch-level augmentation
        resume_from_checkpoint=None,  # Don't resume trainer state - we loaded weights manually
    )
    
    # Run training
    trainer.run()
    
    print(f"💾 Model saved to: {training_args.output_dir}")
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 