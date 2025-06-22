#!/usr/bin/env python3
"""Training script for ImageNet with PReLU CNN"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from prelu_cnn import CNN, CNNTrainer, preprocess_images

def main():
    print("Loading ImageNet dataset...")
    
    # Load ImageNet dataset (this will take a while and requires significant storage)
    # Note: You need to have ImageNet access through Hugging Face
    try:
        # Load a subset for faster experimentation
        dataset = load_dataset("imagenet-1k", split="train[:1%]")  # Use 1% for testing
        eval_dataset = load_dataset("imagenet-1k", split="validation[:10%]")  # Use 10% of validation
        
        print(f"Training samples: {len(dataset)}")
        print(f"Validation samples: {len(eval_dataset)}")
        
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        print("You may need to:")
        print("1. Request access to ImageNet on Hugging Face")
        print("2. Use: huggingface-cli login")
        print("3. Or use a smaller dataset like Food101 for testing")
        return

    # Preprocess datasets
    print("Preprocessing images...")
    dataset = dataset.map(preprocess_images, batched=True, batch_size=100)
    eval_dataset = eval_dataset.map(preprocess_images, batched=True, batch_size=100)
    
    # Set format for PyTorch
    dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    eval_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

    # Create models for comparison
    print("Creating models...")
    
    # ReLU baseline
    model_relu = CNN(use_prelu=False, use_builtin_conv=True)
    
    # PReLU channel-wise
    model_prelu_cw = CNN(use_prelu=True, use_builtin_conv=True, prelu_channel_wise=True)
    
    # PReLU channel-shared  
    model_prelu_cs = CNN(use_prelu=True, use_builtin_conv=True, prelu_channel_wise=False)

    models = {
        "ReLU": model_relu,
        "PReLU_ChannelWise": model_prelu_cw, 
        "PReLU_ChannelShared": model_prelu_cs
    }

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./imagenet_results",
        num_train_epochs=10,
        per_device_train_batch_size=32,  # Adjust based on your GPU memory
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=4,
        fp16=True,  # Use mixed precision for faster training
        gradient_accumulation_steps=2,  # Simulate larger batch size
    )

    # Train each model
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Count PReLU parameters specifically
        if "PReLU" in name:
            prelu_params = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n and any(isinstance(m, torch.nn.PReLU) for m in model.modules()))
            print(f"PReLU parameters: {prelu_params:,}")
        
        # Create trainer
        trainer = CNNTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            tokenizer=None,  # Not needed for vision
        )
        
        # Train
        print(f"Starting training for {name}...")
        train_result = trainer.train()
        
        # Evaluate
        print(f"Evaluating {name}...")
        eval_result = trainer.evaluate()
        
        results[name] = {
            'train_loss': train_result.training_loss,
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_loss': eval_result['eval_loss']
        }
        
        # Save model
        trainer.save_model(f"./imagenet_results/{name}")
        
        print(f"{name} Results:")
        print(f"  Training Loss: {train_result.training_loss:.4f}")
        print(f"  Validation Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"  Validation Loss: {eval_result['eval_loss']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"{name:20} | Accuracy: {result['eval_accuracy']:.4f} | Loss: {result['eval_loss']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['eval_accuracy'])
    print(f"\nBest Model: {best_model[0]} with {best_model[1]['eval_accuracy']:.4f} accuracy")

if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    main() 