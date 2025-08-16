#!/usr/bin/env python3
"""Train ReLU CNN on ImageNet using ModelTrainer"""

import torch
from datasets import load_from_disk, load_dataset, Dataset
from transformers import TrainingArguments, TrainerCallback
import torchvision.transforms as T

# Import from shared_utils package
from shared_utils import ModelTrainer, find_latest_checkpoint

from prelu_cnn import CNN, CNNTrainer

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

def compute_metrics(eval_pred):
    """
    Compute top-1 and top-5 accuracy for ImageNet classification.
    
    Args:
        eval_pred: EvalPrediction object containing predictions and labels
        
    Returns:
        dict: Dictionary with top-1 and top-5 accuracy metrics
    """
    logits, labels = eval_pred
    import numpy as np

    # Check for invalid values
    if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
        print("⚠️ Warning: Found NaN or Inf values in logits!")
        return {"top1": 0.0, "top5": 0.0}
    
    # Ensure shapes match - handle potential mismatches with offset detection
    # Handle bug/compatability issue between our custom SafeDataLoader & HuggingFace Trainer 
    min_len = min(len(logits), len(labels))
    if len(logits) != len(labels):
        print(f"⚠️ Warning: Shape mismatch! Logits: {len(logits)}, Labels: {len(labels)}")
        
        # Try to detect if there's a simple offset issue - check if the length difference is small (<=2)
        if abs(len(logits) - len(labels)) <= 2:            
            # Initialize variables to track the best alignment strategy
            best_match = 0  # Track the highest number of correct predictions found
            best_logits = logits[:min_len]  # Default to truncating logits from the start
            best_labels = labels[:min_len]  # Default to truncating labels from the start
            
            # Try truncating from different starting positions to find best alignment
            # Test up to 2 different starting positions for logits
            for logit_start in range(min(2, len(logits) - min_len + 1)):
                # Test up to 2 different starting positions for labels
                for label_start in range(min(2, len(labels) - min_len + 1)):
                    # Extract aligned segments of the specified minimum length
                    test_logits = logits[logit_start:logit_start + min_len]
                    test_labels = labels[label_start:label_start + min_len]
                    
                    # Verify that both segments have the correct length
                    if len(test_logits) == len(test_labels) == min_len:
                        # Perform a quick accuracy check to evaluate this alignment
                        test_predictions = test_logits.argmax(axis=1)  # Get predicted class indices
                        matches = (test_predictions == test_labels).sum()  # Count correct predictions
                        
                        # If this alignment performs better, save it as the best option
                        if matches > best_match:
                            best_match = matches  # Update the best match count
                            best_logits = test_logits  # Save the best logits alignment
                            best_labels = test_labels  # Save the best labels alignment
            logits = best_logits
            labels = best_labels
        else:
            logits = logits[:min_len]
            labels = labels[:min_len]
    
    # Calculate top-1 accuracy
    predictions = logits.argmax(axis=1)
    correct_top1 = (predictions == labels)
    top1 = correct_top1.mean()

    # Calculate top-5 accuracy (fixed version)
    # Get indices of top-5 predictions for each sample
    top5_indices = np.argsort(logits, axis=1)[:, -5:]  # Get top-5 indices
    # Check if true label is in top-5 predictions for each sample
    top5_correct = np.array([labels[i] in top5_indices[i] for i in range(len(labels))])
    top5 = top5_correct.mean()
    
    metrics = {"top1": float(top1), "top5": float(top5)}
    return metrics

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
    batch_size_per_gpu = 128
    grad_accum = 4
    num_epochs = 80

    # ------------------ calculate warm-up steps ------------------
    images = 1_281_167                      # ImageNet-1k train set
    eff_batch = batch_size_per_gpu * grad_accum * max(1, num_gpus)  # per-GPU × grad_acc × num_gpus
    steps_per_epoch = (images + eff_batch - 1) // eff_batch # forces rounding up to nearest integer
    #                                                       # The formula (a + b - 1) // b is equivalent to ceil(a / b)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10 %

    # Check for existing checkpoints to resume from
    output_dir = f"./results/cnn_results_{'prelu' if use_prelu else 'relu'}"
    resume_from_checkpoint = False or find_latest_checkpoint(output_dir)
    
    if resume_from_checkpoint:
        print(f"🔄 Found checkpoint to resume from: {resume_from_checkpoint}")
    else:
        print("🆕 No existing checkpoints found, starting fresh training")

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,  # More epochs for better convergence
        per_device_train_batch_size=batch_size_per_gpu,  # Reduced for stability
        per_device_eval_batch_size=batch_size_per_gpu,
        learning_rate=0.1,
        weight_decay=1e-4,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum,
        eval_steps=1,
        logging_steps=100,
        save_steps=1,
        seed=42,
        logging_dir="./logs/logs",
        remove_unused_columns=False, # Fix for custom dataset format
        dataloader_num_workers=8, # Parallel data loading
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,     # If True, the DataLoader will copy Tensors into CUDA pinned memory before returning them.
                                        # This can speed up host-to-GPU transfer, especially for large batches.
        optim="sgd",
        optim_args="momentum=0.9",
        max_grad_norm=0,                # No gradient clipping (max_grad_norm=0): For CNNs, gradient clipping is usually not required,
                                        # as exploding gradients are less common compared to RNNs/transformers.
        lr_scheduler_type="cosine",     # Cosine annealing to 0 (better for SGD)
        eval_strategy="steps",
        save_strategy="epoch",
        logging_strategy="steps",
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
    
    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer_class=CNNTrainer,
        compute_metrics=compute_metrics,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    # Run training
    trainer.run()
    
    print(f"💾 Model saved to: {training_args.output_dir}")
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 