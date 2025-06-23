#!/usr/bin/env python3
"""Train ReLU CNN on ImageNet using ModelTrainer"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments

# Import from shared_utils package
from shared_utils import ModelTrainer

from prelu_cnn import CNN, CNNTrainer, preprocess_images

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
    
    # Load ImageNet dataset (subset for faster training)
    # Use subset for faster experimentation
    train_dataset = load_dataset("imagenet-1k", split="train")  # 10k samples
    eval_dataset = load_dataset("imagenet-1k", split="validation[:2000]")  # 2k samples
    
    print(f"✅ Training samples: {len(train_dataset):,}")
    print(f"✅ Validation samples: {len(eval_dataset):,}")
    num_classes = 1000
    
    # Create ReLU CNN model
    print(f"\n🏗️ Creating ReLU CNN model ({num_classes} classes)...")
    model = CNN(
        use_prelu=False,  # Use ReLU
        use_builtin_conv=True,  # Use fast PyTorch convolutions
        num_classes=num_classes
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir="./relu_cnn_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=True,  # Mixed precision for faster training
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=25,
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=42,
        remove_unused_columns=False,
        logging_dir="./logs",
    )
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Evaluation steps: {training_args.eval_steps}")
    print(f"  Output directory: {training_args.output_dir}")
    
    # Create trainer using ModelTrainer
    print(f"\n🏋️ Setting up trainer...")
    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_fn=preprocess_images,
        trainer_class=CNNTrainer
    )
    
    # Run training
    print(f"\n🎯 Starting training...")
    trainer_instance, results = trainer.run()
    
    # Print final results
    print(f"\n{'='*60}")
    print("🏆 TRAINING COMPLETED")
    print(f"{'='*60}")
    
    accuracy = results.get('eval_accuracy', 0.0)
    loss = results.get('eval_loss', float('inf'))
    
    print(f"📈 Final Results:")
    print(f"  🎯 Accuracy: {accuracy:.4f}")
    print(f"  📉 Loss: {loss:.4f}")
    print(f"  💾 Model saved to: {training_args.output_dir}")
    
    # Additional metrics if available
    for key, value in results.items():
        if key not in ['eval_accuracy', 'eval_loss']:
            print(f"  📊 {key}: {value}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Check {training_args.output_dir} for saved model and logs")

if __name__ == "__main__":
    main() 