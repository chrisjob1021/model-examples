#!/usr/bin/env python3
"""Train ReLU CNN on ImageNet using ModelTrainer"""

import torch
from datasets import load_from_disk, load_dataset, Dataset
from transformers import TrainingArguments, TrainerCallback
import torchvision.transforms as T

# Import from shared_utils package
from shared_utils import ModelTrainer

from prelu_cnn import CNN, CNNTrainer

class StepLRSchedulerCallback(TrainerCallback):
    """Custom callback to implement step-wise learning rate decay at specific epochs."""
    
    def __init__(self, decay_epochs=[30, 60, 80], decay_factor=0.1):
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor
        self.original_lr = None
        self.current_decay_level = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Store the original learning rate at the start of training."""
        self.original_lr = args.learning_rate
        print(f"📊 Step LR Scheduler initialized:")
        print(f"   Initial LR: {self.original_lr}")
        print(f"   Decay epochs: {self.decay_epochs}")
        print(f"   Decay factor: {self.decay_factor}")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Check if we need to decay the learning rate at the start of each epoch."""
        current_epoch = state.epoch
        
        # Example values:
        #   self.decay_epochs = [30, 60, 80]
        #   current_epoch = 65
        #   decay_level = sum(1 for epoch in [30, 60, 80] if 65 >= epoch) = 2
        decay_level = sum(1 for epoch in self.decay_epochs if current_epoch >= epoch)
        
        if decay_level > self.current_decay_level:
            self.current_decay_level = decay_level
            new_lr = self.original_lr * (self.decay_factor ** decay_level)
            
            # Update the learning rate in the optimizer
            for param_group in kwargs['optimizer'].param_groups:
                param_group['lr'] = new_lr
                
            print(f"\n📉 Learning rate decayed at epoch {int(current_epoch)}:")
            print(f"   New LR: {new_lr} (decay level: {decay_level})")

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
                    'label': 0
                })
        return batch
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset."""
        return getattr(self.dataset, name)

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
        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),  # Ensure 3 channels (convert grayscale to RGB)
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),      # Randomly crop and resize to 224x224 (simulates zoom/scale)
        T.RandomHorizontalFlip(),                         # Randomly flip images horizontally (augmentation)
        T.RandAugment(num_ops=2, magnitude=9),            # Apply 2 random augmentations with magnitude 9 (extra augmentation)
        T.ToTensor(),                                     # Convert PIL Image or numpy.ndarray to tensor and scale to [0, 1]
        T.Normalize(mean, std),                           # Normalize using ImageNet mean and std
        T.RandomErasing(p=0.25, scale=(0.02, 0.1)),       # Randomly erase a rectangle region (extra augmentation, 25% chance)
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
        #learning_rate=3e-4,
        learning_rate=0.01,  # Start with 0.01 as in PReLU paper
        # The paper used weight_decay=0.0001 (1e-4).However, for deep CNNs without BN, some practitioners use higher values (0.0005).
        weight_decay=0.0005,
        #weight_decay=0.1, # weight_decay=0.05 is common for ViTs, but deep CNNs with no BN weight-decay exemption often work better at 0.1 – 0.15. 
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
        dataloader_persistent_workers=False,
        # If True, the DataLoader will copy Tensors into CUDA pinned memory before returning them.
        # This can speed up host-to-GPU transfer, especially for large batches.
        dataloader_pin_memory=True,
        # This argument, when set to True, ensures that training always starts from the beginning of the dataset,
        # even if a checkpoint is provided. It disables fast-forwarding/skipping of data that would otherwise occur
        # when resuming from a checkpoint, which is useful for reproducibility or when using custom datasets.
        #ignore_data_skip=True,
        # Optimizer and scheduler settings
        optim="sgd",  # SGD optimizer
        optim_args="momentum=0.9",
        # lr_scheduler_type="cosine_with_min_lr",  # Use cosine LR with min_lr_rate
        # lr_scheduler_kwargs={"min_lr_rate": 0.1},  # Not needed for current step-wise scheduler
        #   Added StepLRSchedulerCallback: This custom callback implements the step decay:
        #     - Decays learning rate by 10x at epochs 30, 60, and 80
        #     - So LR progression will be: 0.01 → 0.001 → 0.0001 → 0.00001

        #   The scheduler will:
        #   - Start at LR = 0.01
        #   - At epoch 30: LR = 0.001 (0.01 × 0.1)
        #   - At epoch 60: LR = 0.0001 (0.01 × 0.1²)
        #   - At epoch 80: LR = 0.00001 (0.01 × 0.1³)
        lr_scheduler_type="constant_with_warmup",  # Use constant LR with manual drops
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
        label_names=["labels"], # need this to get eval_loss
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
    
    # Create step LR scheduler callback
    step_lr_callback = StepLRSchedulerCallback(
        decay_epochs=[30, 60, 80],  # Decay at these epochs
        decay_factor=0.1  # Multiply LR by 0.1 at each decay
    )
    
    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer_class=CNNTrainer,
        callbacks=[step_lr_callback],  # Add the step scheduler callback
    )
    
    # Run training
    trainer.run()
    
    print(f"💾 Model saved to: {training_args.output_dir}")
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 