#!/usr/bin/env python3
"""ImageNet trainer for PReLU CNN using shared_utils"""

import torch
import sys
import os
from datasets import load_dataset
from transformers import TrainingArguments

# Add shared_utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
from trainer import ModelTrainer

from prelu_cnn import CNN, CNNTrainer, preprocess_images

class ImageNetCNNTrainer(ModelTrainer):
    """ImageNet trainer for CNN using shared ModelTrainer utilities."""
    
    def __init__(self, model, training_args, train_dataset, eval_dataset):
        super().__init__(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocess_fn=preprocess_images,
            trainer_class=CNNTrainer
        )

def setup_device():
    """Setup and display device information."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Available Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
    
    return device

def load_datasets(train_samples=None, eval_samples=None):
    """Load ImageNet datasets with optional sample limits."""
    print("Loading ImageNet dataset...")
    
    try:
        # Determine split strings
        train_split = f"train[:{train_samples}]" if train_samples else "train"
        eval_split = f"validation[:{eval_samples}]" if eval_samples else "validation"
        
        # Load datasets
        train_dataset = load_dataset("imagenet-1k", split=train_split)
        eval_dataset = load_dataset("imagenet-1k", split=eval_split)
        
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(eval_dataset):,}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        print(f"❌ Error loading ImageNet: {e}")
        print("\nTroubleshooting steps:")
        print("1. Request access to ImageNet on Hugging Face Hub")
        print("2. Run: huggingface-cli login")
        print("3. Accept ImageNet terms of use")
        
        # Fallback to CIFAR-10 for testing
        print(f"\n🔄 Falling back to CIFAR-10 for testing...")
        train_dataset = load_dataset("cifar10", split=f"train[:{train_samples or 5000}]")
        eval_dataset = load_dataset("cifar10", split=f"test[:{eval_samples or 1000}]")
        
        print(f"CIFAR-10 - Training: {len(train_dataset):,}, Validation: {len(eval_dataset):,}")
        return train_dataset, eval_dataset, 10  # Return num_classes for CIFAR-10

def create_training_args(**kwargs):
    """Create training arguments with sensible defaults."""
    
    # Default configuration optimized for ImageNet
    defaults = {
        "output_dir": "./imagenet_results",
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "warmup_steps": 300,
        "fp16": True,  # Mixed precision for faster training
        "gradient_accumulation_steps": 2,
        "dataloader_num_workers": 4,
        "evaluation_strategy": "steps",
        "eval_steps": 250,
        "logging_steps": 50,
        "save_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "seed": 42,
        "remove_unused_columns": False,
        "logging_dir": "./logs",
    }
    
    # Update with any provided overrides
    defaults.update(kwargs)
    
    return TrainingArguments(**defaults)

def compare_relu_vs_prelu(train_dataset, eval_dataset, training_args, num_classes=1000):
    """Compare ReLU baseline vs PReLU variants."""
    
    models = {
        "ReLU_Baseline": CNN(use_prelu=False, use_builtin_conv=True, num_classes=num_classes),
        "PReLU_ChannelWise": CNN(use_prelu=True, use_builtin_conv=True, prelu_channel_wise=True, num_classes=num_classes),
        "PReLU_ChannelShared": CNN(use_prelu=True, use_builtin_conv=True, prelu_channel_wise=False, num_classes=num_classes)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training {model_name}")
        print(f"{'='*60}")
        
        # Model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Count PReLU-specific parameters
        if "PReLU" in model_name:
            prelu_params = sum(
                p.numel() for name, p in model.named_parameters() 
                if 'weight' in name and any('prelu' in str(type(m)).lower() for m in model.modules())
            )
            print(f"  PReLU parameters: {prelu_params:,}")
        
        # Create model-specific training args
        model_training_args = TrainingArguments(
            **{**training_args.to_dict(), "output_dir": f"{training_args.output_dir}/{model_name}"}
        )
        
        # Create and run trainer
        trainer = ImageNetCNNTrainer(
            model=model,
            training_args=model_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        print(f"🏋️ Starting training...")
        trainer_instance, eval_results = trainer.run()
        
        # Store results
        accuracy = eval_results.get('eval_accuracy', 0.0)
        loss = eval_results.get('eval_loss', float('inf'))
        
        results[model_name] = {
            'accuracy': accuracy,
            'loss': loss,
            'total_params': total_params,
            'trainer': trainer_instance
        }
        
        print(f"\n✅ {model_name} Results:")
        print(f"  🎯 Accuracy: {accuracy:.4f}")
        print(f"  📉 Loss: {loss:.4f}")
        print(f"  💾 Model saved to: {model_training_args.output_dir}")
    
    return results

def print_final_comparison(results):
    """Print final comparison table."""
    
    print(f"\n{'='*70}")
    print("🏆 FINAL RESULTS COMPARISON")
    print(f"{'='*70}")
    
    # Header
    print(f"{'Model':<20} {'Accuracy':<10} {'Loss':<10} {'Parameters':<12} {'Δ Params':<10}")
    print("-" * 70)
    
    # Get baseline for comparison
    baseline_params = results['ReLU_Baseline']['total_params']
    
    # Sort by accuracy (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for name, result in sorted_results:
        accuracy = result['accuracy']
        loss = result['loss']
        params = result['total_params']
        
        # Parameter difference from baseline
        param_diff = params - baseline_params
        param_diff_str = "baseline" if param_diff == 0 else f"+{param_diff:,}" if param_diff > 0 else f"{param_diff:,}"
        
        print(f"{name:<20} {accuracy:<10.4f} {loss:<10.4f} {params:<12,} {param_diff_str:<10}")
    
    # Analysis
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🥇 Best Model: {best_model[0]} ({best_model[1]['accuracy']:.4f} accuracy)")
    
    # PReLU analysis
    prelu_results = {k: v for k, v in results.items() if 'PReLU' in k}
    if prelu_results:
        best_prelu = max(prelu_results.items(), key=lambda x: x[1]['accuracy'])
        baseline_acc = results['ReLU_Baseline']['accuracy']
        improvement = best_prelu[1]['accuracy'] - baseline_acc
        
        print(f"\n📈 PReLU Analysis:")
        print(f"  Best PReLU variant: {best_prelu[0]}")
        print(f"  Improvement over ReLU: {improvement:+.4f} ({improvement/baseline_acc*100:+.2f}%)")

def main():
    """Main training function."""
    
    print("🎯 ImageNet PReLU CNN Training")
    print("=" * 50)
    
    # Setup
    device = setup_device()
    
    # Configuration - use smaller subset for faster experimentation
    TRAIN_SAMPLES = 10000  # ~0.8% of ImageNet
    EVAL_SAMPLES = 2000    # ~4% of validation set
    
    # Load datasets
    dataset_result = load_datasets(TRAIN_SAMPLES, EVAL_SAMPLES)
    
    if len(dataset_result) == 3:
        # CIFAR-10 fallback
        train_dataset, eval_dataset, num_classes = dataset_result
        print(f"📝 Using CIFAR-10 fallback ({num_classes} classes)")
    else:
        # ImageNet success
        train_dataset, eval_dataset = dataset_result
        num_classes = 1000
        print(f"📝 Using ImageNet ({num_classes} classes)")
    
    # Create training arguments
    training_args = create_training_args(
        num_train_epochs=3,  # Quick experiment
        per_device_train_batch_size=8,  # Conservative for stability
        eval_steps=100,
        save_steps=300,
    )
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Device: {device}")
    
    # Run comparison
    results = compare_relu_vs_prelu(
        train_dataset, eval_dataset, training_args, num_classes
    )
    
    # Final results
    print_final_comparison(results)
    print(f"\n🎉 Training completed! Results saved to {training_args.output_dir}")

if __name__ == "__main__":
    main() 