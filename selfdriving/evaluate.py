import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import json

from models.trajectory_model import TrajectoryPredictionModel, MultiModalLoss
from data.ngsim_dataset import get_ngsim_dataloaders


def calculate_metrics(predictions, ground_truth):
    
    batch_size, pred_len, num_modes, _ = predictions.shape
    
    gt_expanded = ground_truth.unsqueeze(2).expand(-1, -1, num_modes, -1)
    
    distances = torch.norm(predictions - gt_expanded, dim=-1)
    
    ade_per_mode = distances.mean(dim=1)
    fde_per_mode = distances[:, -1, :]
    
    min_ade = torch.min(ade_per_mode, dim=-1)[0]
    min_fde = torch.min(fde_per_mode, dim=-1)[0]
    
    avg_ade = ade_per_mode.mean(dim=-1)
    avg_fde = fde_per_mode.mean(dim=-1)
    
    return {
        'min_ade': min_ade.mean().item(),
        'min_fde': min_fde.mean().item(),
        'avg_ade': avg_ade.mean().item(),
        'avg_fde': avg_fde.mean().item(),
        'min_ade_all': min_ade.cpu().numpy(),
        'min_fde_all': min_fde.cpu().numpy()
    }


def visualize_predictions(model, dataloader, device, num_samples=5, save_dir='selfdriving/visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    samples_plotted = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_plotted >= num_samples:
                break
            
            hist = batch['hist'].to(device)
            fut = batch['fut'].to(device)
            neighbors = batch['neighbors']
            
            predictions, mode_probs = model(hist, neighbors, pred_len=fut.size(1))
            
            batch_size = min(hist.size(0), num_samples - samples_plotted)
            
            for i in range(batch_size):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                hist_np = hist[i].cpu().numpy()
                fut_np = fut[i].cpu().numpy()
                pred_np = predictions[i].cpu().numpy()
                probs_np = mode_probs[i].cpu().numpy().mean(axis=0)
                
                ax1.plot(hist_np[:, 0], hist_np[:, 1], 'b-', linewidth=2, label='History')
                ax1.plot(fut_np[:, 0], fut_np[:, 1], 'g-', linewidth=2, label='Ground Truth')
                
                colors = plt.cm.rainbow(np.linspace(0, 1, pred_np.shape[1]))
                for mode_idx in range(pred_np.shape[1]):
                    ax1.plot(pred_np[:, mode_idx, 0], pred_np[:, mode_idx, 1],
                            color=colors[mode_idx], alpha=0.5, linewidth=1,
                            label=f'Mode {mode_idx+1} (p={probs_np[mode_idx]:.2f})')
                
                ax1.scatter(hist_np[0, 0], hist_np[0, 1], c='blue', s=100, marker='o', label='Start')
                ax1.scatter(hist_np[-1, 0], hist_np[-1, 1], c='red', s=100, marker='s', label='Current')
                
                ax1.set_xlabel('X position (feet)')
                ax1.set_ylabel('Y position (feet)')
                ax1.set_title('Trajectory Prediction')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
                ax1.axis('equal')
                
                time_steps = np.arange(pred_np.shape[0])
                ax2.bar(np.arange(pred_np.shape[1]), probs_np, color=colors)
                ax2.set_xlabel('Mode')
                ax2.set_ylabel('Average Probability')
                ax2.set_title('Mode Probabilities')
                ax2.set_xticks(np.arange(pred_np.shape[1]))
                ax2.set_xticklabels([f'Mode {i+1}' for i in range(pred_np.shape[1])])
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'prediction_sample_{samples_plotted+1}.png'), dpi=150)
                plt.close()
                
                samples_plotted += 1
    
    print(f"Saved {samples_plotted} visualization samples to {save_dir}")


def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_metrics = {
        'min_ade': [],
        'min_fde': [],
        'avg_ade': [],
        'avg_fde': []
    }
    
    criterion = MultiModalLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        for batch in progress_bar:
            hist = batch['hist'].to(device)
            fut = batch['fut'].to(device)
            neighbors = batch['neighbors']
            
            predictions, mode_probs = model(hist, neighbors, pred_len=fut.size(1))
            
            loss, _ = criterion(predictions, mode_probs, fut)
            total_loss += loss.item()
            
            metrics = calculate_metrics(predictions, fut)
            for key in ['min_ade', 'min_fde', 'avg_ade', 'avg_fde']:
                all_metrics[key].append(metrics[key])
            
            progress_bar.set_postfix({
                'min_ade': metrics['min_ade'],
                'min_fde': metrics['min_fde']
            })
    
    final_metrics = {
        'loss': total_loss / len(dataloader),
        'min_ade': np.mean(all_metrics['min_ade']),
        'min_fde': np.mean(all_metrics['min_fde']),
        'avg_ade': np.mean(all_metrics['avg_ade']),
        'avg_fde': np.mean(all_metrics['avg_fde']),
        'min_ade_std': np.std(all_metrics['min_ade']),
        'min_fde_std': np.std(all_metrics['min_fde'])
    }
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Convolutional Social Pooling Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NGSIM dataset file')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    hist_len = config.get('hist_len', 30)
    pred_len = config.get('pred_len', 50)
    encoder_dim = config.get('encoder_dim', 128)
    num_modes = config.get('num_modes', 6)
    
    print("Loading dataset...")
    _, test_loader = get_ngsim_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        hist_len=hist_len,
        pred_len=pred_len,
        train_split=0.8
    )
    
    print("Initializing model...")
    model = TrajectoryPredictionModel(
        input_dim=2,
        output_dim=2,
        encoder_dim=encoder_dim,
        decoder_dim=encoder_dim,
        num_modes=num_modes
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Min ADE: {metrics['min_ade']:.4f} ± {metrics['min_ade_std']:.4f}")
    print(f"Min FDE: {metrics['min_fde']:.4f} ± {metrics['min_fde_std']:.4f}")
    print(f"Avg ADE: {metrics['avg_ade']:.4f}")
    print(f"Avg FDE: {metrics['avg_fde']:.4f}")
    print("="*50)
    
    results_file = args.checkpoint.replace('.pth', '_eval_results.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    if args.visualize:
        print(f"\nGenerating {args.num_samples} visualization samples...")
        visualize_predictions(model, test_loader, device, num_samples=args.num_samples)


if __name__ == '__main__':
    main()