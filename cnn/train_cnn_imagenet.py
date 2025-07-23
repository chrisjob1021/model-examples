#!/usr/bin/env python3
"""Train ReLU CNN on ImageNet using ModelTrainer"""

import torch
from datasets import load_from_disk
from transformers import TrainingArguments

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
    
    # Load preprocessed ImageNet dataset from disk
    print("🔄 Loading preprocessed datasets from disk...")
    
    # Load training dataset
    dataset_path = "./processed_datasets/imagenet_processor"
    dataset = load_from_disk(dataset_path)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"].select(range(1000))
    
    print(f"✅ Loaded preprocessed datasets from disk")
    print(f"✅ Training samples: {len(train_dataset):,}")
    print(f"✅ Validation samples: {len(eval_dataset):,}")
    
    # Display dataset info for debugging
    print(f"📊 Dataset Info:")
    print(f"  Train dataset features: {list(train_dataset.features.keys())}")
    print(f"  Train sample keys: {list(train_dataset[0].keys())}")
    print(f"  Train sample pixel_values shape: {len(train_dataset[0]['pixel_values'])}")
    print(f"  Train sample labels: {train_dataset[0]['labels']}")
    
    use_prelu = False
    
    # Create ReLU CNN model
    print(f"\n🏗️ Creating ReLU CNN model ({1000} classes)...")
    model = CNN(
        use_prelu=use_prelu,  # Use ReLU
        use_builtin_conv=True,  # Use fast PyTorch convolutions
        num_classes=1000
    )
    
    # Move model to GPU
    model = model.to(device)
    print(f"✅ Model moved to device: {device}")
    
    # Verify model is on GPU
    if next(model.parameters()).device != device:
        print(f"❌ Warning: Model parameters are on {next(model.parameters()).device}, expected {device}")
    else:
        print(f"✅ Model parameters confirmed on {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/cnn_results_{'prelu' if use_prelu else 'relu'}",
        num_train_epochs=10,  # More epochs for better convergence
        per_device_train_batch_size=64,  # Reduced for stability
        per_device_eval_batch_size=64,
        learning_rate=1e-3,  # Much more reasonable learning rate
        weight_decay=1e-4,  # Reduced weight decay
        warmup_steps=1000,  # Warmup for better training stability
        gradient_accumulation_steps=4,  # Reduced for more frequent updates
        eval_steps=50,  # Less frequent evaluation
        logging_steps=25,  # Less frequent logging
        save_steps=500,  # Save less frequently
        seed=42,
        logging_dir="./logs/logs",
        # Fix for custom dataset format
        remove_unused_columns=False,
        # Parallel data loading
        dataloader_num_workers=4,  # Reduced to avoid memory issues
        # Optimizer and scheduler settings
        optim="adamw_torch",  # Explicit optimizer
        lr_scheduler_type="cosine",  # Cosine annealing scheduler
        max_grad_norm=1.0,  # Gradient clipping
        # Evaluation settings
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        # Model saving
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    print(f"\n🎯 Starting training...")
    # _, results = trainer.run()
    
    # accuracy = results.get('eval_accuracy', 0.0)
    # loss = results.get('eval_loss', float('inf'))

    # Evaluate the model after training
    eval_results = trainer.evaluate()
    accuracy = eval_results.get('eval_accuracy', 0.0)
    loss = eval_results.get('eval_loss', float('inf'))
    
    print(f"\n📊 Final Results:")
    print(f"  Evaluation Accuracy: {accuracy:.4f}")
    print(f"  Evaluation Loss: {loss:.4f}")
    
    print(f"💾 Model saved to: {training_args.output_dir}")
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 