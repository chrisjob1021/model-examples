<<<<<<< Updated upstream
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
from tqdm import tqdm
import json
from datetime import datetime

from models.trajectory_model import TrajectoryPredictionModel, MultiModalLoss
from data.ngsim_dataset import get_ngsim_dataloaders


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    metrics_sum = {'classification_loss': 0.0, 'regression_loss': 0.0, 
                   'min_ade': 0.0, 'min_fde': 0.0}
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        hist = batch['hist'].to(device)
        fut = batch['fut'].to(device)
        neighbors = batch['neighbors']
        
        optimizer.zero_grad()
        
        predictions, mode_probs = model(hist, neighbors, pred_len=fut.size(1))
        
        loss, metrics = criterion(predictions, mode_probs, fut)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key in metrics_sum:
            metrics_sum[key] += metrics[key]
        
        progress_bar.set_postfix({'loss': loss.item(), 'ade': metrics['min_ade']})
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    
    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics_sum = {'classification_loss': 0.0, 'regression_loss': 0.0,
                   'min_ade': 0.0, 'min_fde': 0.0}
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for batch in progress_bar:
            hist = batch['hist'].to(device)
            fut = batch['fut'].to(device)
            neighbors = batch['neighbors']
            
            predictions, mode_probs = model(hist, neighbors, pred_len=fut.size(1))
            
            loss, metrics = criterion(predictions, mode_probs, fut)
            
            total_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
            
            progress_bar.set_postfix({'loss': loss.item(), 'ade': metrics['min_ade']})
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Convolutional Social Pooling Model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NGSIM dataset file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--encoder_dim', type=int, default=128,
                       help='Hidden dimension for encoder')
    parser.add_argument('--num_modes', type=int, default=6,
                       help='Number of prediction modes')
    parser.add_argument('--hist_len', type=int, default=30,
                       help='Length of history trajectory (in frames)')
    parser.add_argument('--pred_len', type=int, default=50,
                       help='Length of predicted trajectory (in frames)')
    parser.add_argument('--save_dir', type=str, default='selfdriving/checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='selfdriving/logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.log_dir, f'run_{timestamp}'))
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_loader, val_loader = get_ngsim_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        hist_len=args.hist_len,
        pred_len=args.pred_len
    )
    
    print("Initializing model...")
    model = TrajectoryPredictionModel(
        input_dim=2,
        output_dim=2,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.encoder_dim,
        num_modes=args.num_modes
    ).to(device)
    
    criterion = MultiModalLoss(num_modes=args.num_modes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    config = vars(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train ADE: {train_metrics['min_ade']:.4f}, Val ADE: {val_metrics['min_ade']:.4f}")
        print(f"Train FDE: {train_metrics['min_fde']:.4f}, Val FDE: {val_metrics['min_fde']:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/train_ade', train_metrics['min_ade'], epoch)
        writer.add_scalar('Metrics/val_ade', val_metrics['min_ade'], epoch)
        writer.add_scalar('Metrics/train_fde', train_metrics['min_fde'], epoch)
        writer.add_scalar('Metrics/val_fde', val_metrics['min_fde'], epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
=======
#!/usr/bin/env python3
"""Train Convolutional Social Pooling model on Waymo dataset using HuggingFace Transformers."""

import torch
import torch.nn as nn
import os
import yaml
from pathlib import Path
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np

# Import from shared_utils package
import sys
sys.path.append('..')
from shared_utils import ModelTrainer, find_latest_checkpoint

# Import model and data loader
from models.csp_model import ConvSocialPoolingModel
from data.waymo_loader import WaymoTrajectoryDataset


class CSPTrainer(Trainer):
    """Custom trainer for Convolutional Social Pooling model."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for trajectory prediction.
        
        Combines:
        - Negative Log-Likelihood (NLL) loss for trajectory prediction
        - Cross-entropy loss for mode selection
        """
        # Extract inputs
        history = inputs['history']
        social_tensor = inputs['social_tensor']
        future_positions = inputs['future'][:, :, :2]  # Get only x,y positions from future
        
        # Forward pass
        predictions, mode_probs = model(history, social_tensor)
        # predictions: [batch, future_len, num_modes, 2]
        # mode_probs: [batch, num_modes]
        
        batch_size = predictions.size(0)
        future_len = predictions.size(1)
        num_modes = predictions.size(2)
        
        # Compute NLL loss for each mode
        # Reshape for easier computation
        pred_reshaped = predictions.view(batch_size, future_len, num_modes, 2)
        gt_reshaped = future_positions.unsqueeze(2).expand(-1, -1, num_modes, -1)
        
        # L2 distance for each mode
        distances = torch.norm(pred_reshaped - gt_reshaped, dim=-1)  # [batch, future_len, num_modes]
        
        # Sum over time for each mode
        mode_losses = distances.sum(dim=1)  # [batch, num_modes]
        
        # Find best mode (minimum loss)
        best_mode_losses, best_mode_indices = mode_losses.min(dim=1)
        
        # NLL loss: negative log probability of best mode
        nll_loss = -torch.log(mode_probs.gather(1, best_mode_indices.unsqueeze(1)) + 1e-8).squeeze()
        
        # Trajectory loss: L2 distance of best mode
        trajectory_loss = best_mode_losses
        
        # Get loss weights from config
        trajectory_weight = self.args.trajectory_weight if hasattr(self.args, 'trajectory_weight') else 1.0
        nll_weight = self.args.nll_weight if hasattr(self.args, 'nll_weight') else 0.5
        
        # Combined loss
        loss = trajectory_weight * trajectory_loss.mean() + nll_weight * nll_loss.mean()
        
        return (loss, predictions) if return_outputs else loss


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_collator(config):
    """Create a data collator function for batching."""
    def collate_fn(batch):
        """Collate batch of samples."""
        # Stack tensors for each field
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        return collated
    return collate_fn


def main():
    """Main training function."""
    
    print("🚀 Training Convolutional Social Pooling Model on Waymo Dataset")
    print("=" * 60)
    
    # Load configuration
    config_path = Path("configs/default_config.yaml")
    print(f"📋 Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Extract configuration sections
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load datasets
    print("\n📊 Loading Waymo dataset...")
    waymo_path = os.path.expanduser(data_config['waymo_path'])
    
    train_dataset = WaymoTrajectoryDataset(
        data_path=waymo_path,
        split='training',
        config=config
    )
    
    eval_dataset = WaymoTrajectoryDataset(
        data_path=waymo_path,
        split='testing',  # Using testing split for validation
        config=config
    )
    
    print(f"✅ Training samples: {len(train_dataset):,}")
    print(f"✅ Validation samples: {len(eval_dataset):,}")
    
    # Create model
    print(f"\n🏗️ Creating Convolutional Social Pooling model...")
    model = ConvSocialPoolingModel(config)
    
    # Move model to device
    model = model.to(device)
    print(f"✅ Model moved to {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Hidden dimension: {model_config['hidden_dim']}")
    print(f"  Number of LSTM layers: {model_config['num_layers']}")
    print(f"  Number of prediction modes: {model_config['num_modes']}")
    
    # Setup output directory
    output_dir = Path("./results/csp_waymo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for resume
    resume_from = training_config.get('resume_from', None)
    if resume_from:
        checkpoint_path = find_latest_checkpoint(str(output_dir))
        if checkpoint_path:
            print(f"\n🔄 RESUME MODE: Found checkpoint at {checkpoint_path}")
            # Load model weights
            checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(checkpoint_file):
                state_dict = torch.load(checkpoint_file, map_location=device)
                model.load_state_dict(state_dict)
                print(f"✅ Model weights loaded successfully")
        else:
            print(f"⚠️ No checkpoint found, starting fresh training")
            resume_from = None
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=data_config['batch_size'],
        per_device_eval_batch_size=data_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        
        # Learning rate schedule
        lr_scheduler_type=training_config['lr_scheduler'],
        warmup_steps=int(len(train_dataset) / data_config['batch_size'] * training_config['lr_warmup_epochs']),
        
        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=training_config['checkpoint_interval'],
        save_total_limit=training_config['keep_best_k'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_dir="./logs/tensorboard" if training_config.get('tensorboard', True) else None,
        logging_steps=training_config.get('log_interval', 10),
        report_to="tensorboard" if training_config.get('tensorboard', True) else "none",
        
        # DataLoader settings
        dataloader_num_workers=data_config['num_workers'],
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        
        # Other settings
        seed=42,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        optim="adamw",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        
        # Disable unused column removal since we have custom data format
        remove_unused_columns=False,
    )
    
    # Add custom loss weights to training args
    training_args.trajectory_weight = training_config.get('trajectory_weight', 1.0)
    training_args.nll_weight = training_config.get('nll_weight', 0.5)
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  LR scheduler: {training_args.lr_scheduler_type}")
    print(f"  Warmup epochs: {training_config['lr_warmup_epochs']}")
    print(f"  Weight decay: {training_args.weight_decay}")
    print(f"  Trajectory loss weight: {training_args.trajectory_weight}")
    print(f"  NLL loss weight: {training_args.nll_weight}")
    print(f"  Mixed precision (FP16): {training_args.fp16}")
    print(f"  Output directory: {training_args.output_dir}")
    
    # Create data collator
    data_collator = create_data_collator(config)
    
    # Create trainer using ModelTrainer wrapper
    print(f"\n🏋️ Setting up trainer...")
    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        trainer_class=CSPTrainer,
        resume_from_checkpoint=resume_from,
        disable_timestamped_logging=False,
    )
    
    # Run training
    print(f"\n🎯 Starting training...")
    trainer_instance, train_results, eval_results = trainer.run()
    
    # Save final model
    final_model_path = output_dir / "final_model"
    final_model_path.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving final model to: {final_model_path}")
    torch.save(model.state_dict(), final_model_path / "pytorch_model.bin")
    
    # Save configuration with the model
    with open(final_model_path / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n📈 Training Results:")
    print(f"  Final training loss: {train_results.training_loss:.4f}")
    print(f"\n📊 Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"   Model saved to: {final_model_path}")
    print(f"   Checkpoints saved to: {output_dir}")
    
    # Print instructions for inference
    print(f"\n📝 To use the trained model for inference:")
    print(f"   1. Load the model: model = ConvSocialPoolingModel(config)")
    print(f"   2. Load weights: model.load_state_dict(torch.load('{final_model_path}/pytorch_model.bin'))")
    print(f"   3. Use model.inference(history, social_tensor) for predictions")


if __name__ == "__main__":
>>>>>>> Stashed changes
    main()