#!/usr/bin/env python3
"""Evaluate CNN model on ImageNet validation set.

This script loads a trained CNN checkpoint and evaluates its performance
on the ImageNet validation dataset, calculating top-1 and top-5 accuracy.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
import torchvision.transforms as T
from tqdm import tqdm
import os
import sys
from typing import Tuple
import numpy as np
from PIL import Image

# Add parent directory to path to import shared_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import find_latest_checkpoint
from prelu_cnn import CNN


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Tuple[float, ...]:
    """Calculate top-k accuracy for model predictions.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model output logits of shape (batch_size, num_classes)
    labels : torch.Tensor
        Ground truth labels of shape (batch_size,)
    topk : tuple of int
        Values of k for top-k accuracy calculation
        
    Returns
    -------
    tuple of float
        Top-k accuracies as percentages for each k value
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        
        # Get top-k predictions
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # Transpose to shape (maxk, batch_size)
        
        # Compare predictions with ground truth
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            # Count correct predictions in top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Calculate accuracy percentage
            accuracy = correct_k.mul_(100.0 / batch_size)
            res.append(accuracy.item())
            
        return tuple(res)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, float, float]:
    """Evaluate model on validation dataset.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN model
    dataloader : DataLoader
        Validation data loader
    device : torch.device
        Device to run evaluation on
    verbose : bool
        Whether to show progress bar
        
    Returns
    -------
    tuple of float
        (top1_accuracy, top5_accuracy, average_loss)
    """
    model.eval()
    
    total_top1 = 0.0
    total_top5 = 0.0
    total_loss = 0.0
    total_samples = 0
    
    # Setup progress bar
    if verbose:
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for batch in pbar:
            # Handle different input formats
            if 'pixel_values' in batch:
                inputs = batch['pixel_values']
                labels = batch.get('labels', batch.get('label'))
            elif 'image' in batch:
                inputs = batch['image']
                labels = batch.get('labels', batch.get('label'))
            else:
                raise ValueError("Batch must contain 'pixel_values' or 'image' key")
            
            # Move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracies
            top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            
            batch_size = inputs.size(0)
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'Top-1': f'{total_top1/total_samples:.2f}%',
                    'Top-5': f'{total_top5/total_samples:.2f}%',
                    'Loss': f'{total_loss/total_samples:.4f}'
                })
    
    # Calculate final metrics
    avg_top1 = total_top1 / total_samples
    avg_top5 = total_top5 / total_samples
    avg_loss = total_loss / total_samples
    
    return avg_top1, avg_top5, avg_loss


def get_eval_transform():
    """Get evaluation preprocessing transform for ImageNet."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return T.Compose([
        T.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'mode') and x.mode != 'RGB' else x),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def transform_batch(examples):
    """Transform batch of examples for evaluation."""
    transform = get_eval_transform()
    
    # Handle both single examples and batches
    if isinstance(examples['image'], list):
        examples["pixel_values"] = [transform(image) for image in examples["image"]]
    else:
        examples["pixel_values"] = transform(examples["image"])
    
    # Fix label key for HuggingFace compatibility
    if 'label' in examples:
        examples["labels"] = examples["label"]
        del examples["label"]
    
    del examples["image"]
    return examples


def main():
    """Main evaluation function."""
    # Configuration
    results_dir = '/home/chrisobrien/model-examples/results/cnn_results_prelu'
    batch_size = 128
    num_workers = 4
    num_classes = 1000
    limit_batches = None  # Set to None for full evaluation, or a number for quick testing
    
    # Print header
    print("=" * 60)
    print("🎯 CNN Model Evaluation on ImageNet")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📍 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Find latest checkpoint in results directory
    checkpoint_path = find_latest_checkpoint(results_dir)
    if not checkpoint_path:
        print(f"❌ No checkpoints found in {results_dir}")
        sys.exit(1)
    
    print(f"\n📂 Checkpoint: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load trainer state to get training info
    trainer_state_path = os.path.join(checkpoint_path, 'trainer_state.json')
    if os.path.exists(trainer_state_path):
        import json
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
            print(f"   Epoch: {trainer_state.get('epoch', 'N/A')}")
            print(f"   Global Step: {trainer_state.get('global_step', 'N/A'):,}")
            if 'best_metric' in trainer_state:
                print(f"   Best Eval Loss: {trainer_state['best_metric']:.4f}")
    
    # Determine activation type by checking the checkpoint for PReLU weights
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_path):
        from safetensors import safe_open
        with safe_open(model_path, framework="pt", device="cpu") as f:
            # Check if any PReLU weight keys exist in the checkpoint
            keys = f.keys()
            use_prelu = any('.act.weight' in key for key in keys)
    else:
        # Fallback to directory name if checkpoint not found
        if 'relu' in results_dir.lower():
            use_prelu = False
        elif 'prelu' in results_dir.lower():
            use_prelu = True
        else:
            use_prelu = True  # Default to PReLU
    
    print(f"\n🏗️ Loading Model")
    print(f"   Architecture: CNN")
    print(f"   Activation: {'PReLU' if use_prelu else 'ReLU'}")
    print(f"   Classes: {num_classes}")
    
    # Load model
    try:
        model = CNN.from_pretrained(
            checkpoint_path,
            use_prelu=use_prelu,
            use_builtin_conv=True,
            prelu_channel_wise=True,
            num_classes=num_classes,
            device=device
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Load dataset
    print(f"\n📊 Loading Validation Dataset")
    print(f"   Source: HuggingFace Hub (imagenet-1k)")
    try:
        # Use specific revision to avoid re-download (same as training script)
        eval_dataset = load_dataset("imagenet-1k", split="validation", revision="1.0.0")
    except Exception as e:
        print(f"❌ Failed to load dataset from HuggingFace: {e}")
        sys.exit(1)
    
    # Apply transforms
    eval_dataset = eval_dataset.with_transform(transform_batch)
    print(f"✅ Loaded {len(eval_dataset):,} validation samples")
    
    # Create DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    # Limit batches if specified (for testing)
    if limit_batches:
        print(f"⚠️  Limiting evaluation to {limit_batches} batches")
        from itertools import islice
        dataloader = islice(dataloader, limit_batches)
    
    # Run evaluation
    print(f"\n🚀 Starting Evaluation")
    print(f"   Batch Size: {batch_size}")
    print(f"   Workers: {num_workers}")
    print("-" * 60)
    
    try:
        top1_acc, top5_acc, avg_loss = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    print(f"✨ Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"✨ Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"📉 Average Loss: {avg_loss:.4f}")
    print("=" * 60)
    
    # Calculate error rates
    top1_error = 100 - top1_acc
    top5_error = 100 - top5_acc
    print(f"\n📊 Error Rates:")
    print(f"   Top-1 Error: {top1_error:.2f}%")
    print(f"   Top-5 Error: {top5_error:.2f}%")
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
