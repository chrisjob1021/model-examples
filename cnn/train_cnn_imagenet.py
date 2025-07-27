#!/usr/bin/env python3
"""Train ReLU CNN on ImageNet using ModelTrainer"""

import torch
from datasets import load_from_disk, load_dataset
from transformers import TrainingArguments
import torchvision.transforms as T

# Import from shared_utils package
from shared_utils import ModelTrainer

from prelu_cnn import CNN, CNNTrainer

def main():
    """Train ReLU CNN on ImageNet."""
    
    print("🚀 Training ReLU CNN on ImageNet")
    print("=" * 50)
    
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
        train_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.08, 1.0)),      # Randomly crop and resize to 224x224 (simulates zoom/scale)
            T.RandomHorizontalFlip(),                         # Randomly flip images horizontally (augmentation)
            T.RandAugment(num_ops=2, magnitude=9),            # Apply 2 random augmentations with magnitude 9 (extra augmentation)
            T.ToTensor(),                                     # Convert PIL Image or numpy.ndarray to tensor and scale to [0, 1]
            T.Normalize(mean, std),                           # Normalize using ImageNet mean and std
            T.RandomErasing(p=0.25, scale=(0.02, 0.1)),       # Randomly erase a rectangle region (extra augmentation, 25% chance)
        ])
        
        # Define the preprocessing pipeline for evaluation images (no heavy augmentation)
        eval_transform = T.Compose([
            T.Resize(256),                                    # Resize shorter side to 256 pixels
            T.CenterCrop(224),                                # Crop the center 224x224 region
            T.ToTensor(),                                     # Convert to tensor and scale to [0, 1]
            T.Normalize(mean, std),                           # Normalize using ImageNet mean and std
        ])

        def train_transform_fn(example):
            img = example["image"].convert("RGB")
            example["pixel_values"] = train_transform(img)
            return example
        
        def eval_transform_fn(example):
            img = example["image"].convert("RGB")
            example["pixel_values"] = eval_transform(img)
            return example
        
        train_dataset = train_dataset.with_transform(train_transform_fn)
        eval_dataset = eval_dataset.with_transform(eval_transform_fn)
    
    print(f"✅ Loaded preprocessed datasets from disk")
    print(f"✅ Training samples: {len(train_dataset):,}")
    print(f"✅ Validation samples: {len(eval_dataset):,}")
    
    use_prelu = False
    
    # Create ReLU CNN model
    print(f"\n🏗️ Creating ReLU CNN model ({1000} classes)...")
    model = CNN(
        use_prelu=use_prelu,
        use_builtin_conv=True,  # Use fast PyTorch convolutions
        num_classes=1000
    )
    
    # Force fresh weight initialization to ensure clean start
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        elif hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
    if True:
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
    
    num_gpus = torch.cuda.device_count()
    batch_size_per_gpu = 64
    grad_accum = 4
    num_epochs = 90

    # ------------------ calculate warm-up steps ------------------
    images = 1_281_167                      # ImageNet-1k train set
    eff_batch = batch_size_per_gpu * grad_accum * max(1, num_gpus)  # per-GPU × grad_acc × num_gpus
    steps_per_epoch = (images + eff_batch - 1) // eff_batch # forces rounding up to nearest integer
    #                                                       # The formula (a + b - 1) // b is equivalent to ceil(a / b)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.05 * total_steps)  # 5 %

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/cnn_results_{'prelu' if use_prelu else 'relu'}",
        num_train_epochs=num_epochs,  # More epochs for better convergence
        per_device_train_batch_size=batch_size_per_gpu,  # Reduced for stability
        per_device_eval_batch_size=batch_size_per_gpu,
        #learning_rate=1e-3,
        learning_rate=3e-4,
        #weight_decay=1e-4,
        weight_decay=0.1, # weight_decay=0.05 is common for ViTs, but deep CNNs with no BN weight-decay exemption often work better at 0.1 – 0.15. 
        # warmup_steps=1000,  # Warmup for better training stability
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum,  # Reduced for more frequent updates
        eval_steps=1,
        logging_steps=100,
        save_steps=1,
        seed=42,
        logging_dir="./logs/logs",
        # Fix for custom dataset format
        remove_unused_columns=False,
        # Parallel data loading
        dataloader_num_workers=8,
        # Optimizer and scheduler settings
        optim="adamw_torch",  # Explicit optimizer
        # optim_args="momentum=0.9",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_ratio": 0.1},  # 10% of base LR as minimum
        #max_grad_norm=1.0,  # Gradient clipping
        max_grad_norm=0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        # Model saving
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=False,  # Don't load previous checkpoints
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=False,
        label_names=["label"], # need this to get eval_loss
        report_to="tensorboard",
    )

    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Weight decay: {training_args.weight_decay}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    print(f"  LR scheduler: {training_args.lr_scheduler_type}")
    print(f"  Optimizer: {training_args.optim}")
    print(f"  Gradient clipping: {training_args.max_grad_norm}")
    print(f"  Evaluation steps: {training_args.eval_steps}")
    print(f"  Output directory: {training_args.output_dir}")
    print(f"  Remove unused columns: {training_args.remove_unused_columns}")
    print(f"  Dataloader workers: {training_args.dataloader_num_workers}")
    
    # Create trainer using ModelTrainer
    print(f"\n🏋️ Setting up trainer...")
    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer_class=CNNTrainer,
    )
    
    # Run training
    trainer.run()
    
    print(f"💾 Model saved to: {training_args.output_dir}")
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 