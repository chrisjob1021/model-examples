import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import json
from typing import Dict, Any

# Add parent directory to path to import shared_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils.trainer import ModelTrainer
from selfdriving.models.trajectory_model import TrajectoryPredictionModel, MultiModalLoss
from selfdriving.data.ngsim_dataset import get_ngsim_dataloaders


class TrajectoryTrainer(Trainer):
    """
    Custom Trainer class for trajectory prediction that handles
    the multi-modal loss and custom forward pass.
    """
    
    def __init__(self, criterion=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss for trajectory prediction.
        
        Args:
            model: The trajectory prediction model
            inputs: Dictionary containing hist, fut, and neighbors
            return_outputs: Whether to return outputs along with loss
        """
        hist = inputs['hist']
        fut = inputs['fut']
        neighbors = inputs['neighbors']
        
        # Forward pass
        predictions, mode_probs = model(hist, neighbors, pred_len=fut.size(1))
        
        # Compute loss
        loss, metrics = self.criterion(predictions, mode_probs, fut)
        
        # Log metrics
        if self.state.global_step > 0:
            for key, value in metrics.items():
                self.log({f"train_{key}": value})
        
        if return_outputs:
            return loss, (predictions, mode_probs, metrics)
        return loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, 
                       ignore_keys=None, metric_key_prefix="eval"):
        """
        Custom evaluation loop to handle trajectory metrics.
        """
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        total_loss = 0.0
        metrics_sum = {'classification_loss': 0.0, 'regression_loss': 0.0,
                      'min_ade': 0.0, 'min_fde': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for inputs in dataloader:
                # Move inputs to device
                inputs = self._prepare_inputs(inputs)
                
                # Forward pass
                loss, (predictions, mode_probs, metrics) = self.compute_loss(
                    model, inputs, return_outputs=True
                )
                
                total_loss += loss.item()
                for key in metrics_sum:
                    metrics_sum[key] += metrics[key]
                num_batches += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {f"{metric_key_prefix}_{k}": v / num_batches 
                      for k, v in metrics_sum.items()}
        avg_metrics[f"{metric_key_prefix}_loss"] = avg_loss
        
        # Log metrics
        self.log(avg_metrics)
        
        return avg_metrics


def collate_fn(batch):
    """
    Custom collate function to handle neighbor data.
    """
    hist = torch.stack([item['hist'] for item in batch])
    fut = torch.stack([item['fut'] for item in batch])
    neighbors = [item['neighbors'] for item in batch]
    
    return {
        'hist': hist,
        'fut': fut,
        'neighbors': neighbors
    }


def main():
    parser = argparse.ArgumentParser(description='Train Trajectory Model with HuggingFace Trainer')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to local NGSIM dataset (optional, will try HuggingFace first)')
    parser.add_argument('--dataset_name', type=str, default='ngsim',
                       help='Name of dataset on HuggingFace')
    parser.add_argument('--hist_len', type=int, default=30,
                       help='Length of history trajectory')
    parser.add_argument('--pred_len', type=int, default=50,
                       help='Length of predicted trajectory')
    
    # Model arguments
    parser.add_argument('--encoder_dim', type=int, default=128,
                       help='Hidden dimension for encoder')
    parser.add_argument('--num_modes', type=int, default=6,
                       help='Number of prediction modes')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='selfdriving/outputs',
                       help='Directory to save model outputs')
    parser.add_argument('--num_train_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32,
                       help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32,
                       help='Evaluation batch size per device')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader = get_ngsim_dataloaders(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        batch_size=args.per_device_train_batch_size,
        hist_len=args.hist_len,
        pred_len=args.pred_len
    )
    
    # Initialize model
    print("Initializing model...")
    model = TrajectoryPredictionModel(
        input_dim=2,
        output_dim=2,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.encoder_dim,
        num_modes=args.num_modes
    )
    
    # Initialize loss
    criterion = MultiModalLoss(num_modes=args.num_modes)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_min_ade",
        greater_is_better=False,
        seed=args.seed,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["tensorboard"],
        push_to_hub=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize ModelTrainer with custom TrajectoryTrainer
    model_trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        trainer_class=TrajectoryTrainer,
        data_collator=collate_fn,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Add criterion to trainer kwargs
    model_trainer.trainer_class = lambda **kwargs: TrajectoryTrainer(
        criterion=criterion,
        **kwargs
    )
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training
    print("Starting training with HuggingFace Trainer...")
    trainer, train_results, eval_results = model_trainer.run()
    
    # Save final model
    trainer.save_model(os.path.join(args.output_dir, 'final_model'))
    
    print("\nTraining completed!")
    print(f"Final evaluation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    return trainer, train_results, eval_results


if __name__ == '__main__':
    main()