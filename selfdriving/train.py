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
    main()