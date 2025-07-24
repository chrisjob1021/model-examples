#!/usr/bin/env python3
"""
Visualize CNN layer activations for a single example.
This script loads the trained model and shows what each layer learns.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datasets import load_from_disk
from prelu_cnn import CNN
import os
from pathlib import Path

class ActivationVisualizer:
    """Visualizes activations from each layer of a CNN model."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.activations = {}
        self.layer_names = []
        self.hooks = []
        
        # Register hooks for all convolutional and linear layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
                # Store activation on CPU to save GPU memory
                self.activations[name] = output.detach().cpu()
            return hook
        
        # Hook into conv layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.PReLU)):
                # Skip if it's inside a Sequential that we'll hook separately
                if '.' in name and any(seq_name in name for seq_name in ['conv1', 'conv2', 'conv3', 'classifier']):
                    continue
                    
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
                self.layer_names.append(name)
        
        # Also hook the main sequential blocks
        for block_name in ['conv1', 'conv2', 'conv3']:
            if hasattr(self.model, block_name):
                block = getattr(self.model, block_name)
                hook = block.register_forward_hook(get_activation(block_name))
                self.hooks.append(hook)
                self.layer_names.append(block_name)
        
        # Hook classifier
        if hasattr(self.model, 'classifier'):
            hook = self.model.classifier.register_forward_hook(get_activation('classifier_output'))
            self.hooks.append(hook)
            self.layer_names.append('classifier_output')
    
    def visualize_sample(self, image_tensor, label=None, save_path=None):
        """
        Visualize activations for a single image.
        
        Args:
            image_tensor: Input tensor of shape (1, 3, 224, 224) or (3, 224, 224)
            label: Optional ground truth label
            save_path: Optional path to save the visualization
        """
        # Ensure correct input shape
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        image_tensor = image_tensor.to(self.device)
        
        # Clear previous activations
        self.activations.clear()
        
        # Forward pass to collect activations
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        
        # Create comprehensive visualization
        self._create_visualization(image_tensor, label, predicted_class, confidence, save_path)
    
    def _create_visualization(self, input_image, true_label, predicted_class, confidence, save_path):
        """Create a comprehensive activation visualization."""
        
        # Calculate subplot layout
        conv_layers = [name for name in self.layer_names if 'conv' in name.lower()]
        n_conv = len(conv_layers)
        
        # Create larger figure to accommodate all feature maps
        fig_height = max(6 * n_conv + 4, 20)  # Ensure minimum height for visibility
        fig = plt.figure(figsize=(24, fig_height))
        gs = GridSpec(n_conv + 2, 4, figure=fig, hspace=0.4, wspace=0.15)
        
        # Plot original image
        ax_orig = fig.add_subplot(gs[0, :2])
        self._plot_input_image(ax_orig, input_image, true_label, predicted_class, confidence)
        
        # Plot prediction info
        ax_pred = fig.add_subplot(gs[0, 2:])
        self._plot_prediction_info(ax_pred, predicted_class, confidence, true_label)
        
        # Plot activations for each convolutional layer
        for i, layer_name in enumerate(conv_layers):
            if layer_name in self.activations:
                row = i + 1
                activation = self.activations[layer_name]
                
                # Feature maps overview
                ax_overview = fig.add_subplot(gs[row, 0])
                self._plot_activation_overview(ax_overview, activation, layer_name)
                
                # Individual feature maps
                ax_features = fig.add_subplot(gs[row, 1:])
                self._plot_feature_maps(ax_features, activation, layer_name)
        
        # Plot classifier activations if available
        if 'classifier_output' in self.activations:
            row = len(conv_layers) + 1
            ax_classifier = fig.add_subplot(gs[row, :])
            self._plot_classifier_activations(ax_classifier, self.activations['classifier_output'])
        
        plt.suptitle('CNN Layer Activations Visualization', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def _plot_input_image(self, ax, input_tensor, true_label, predicted_class, confidence):
        """Plot the original input image."""
        # Convert from tensor to displayable format
        img = input_tensor.squeeze(0).cpu()  # Remove batch dim
        
        # Denormalize if needed (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to HWC format for matplotlib
        img = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img)
        ax.set_title(f'Input Image\nTrue: {true_label}, Pred: {predicted_class}\nConfidence: {confidence:.3f}')
        ax.axis('off')
    
    def _plot_prediction_info(self, ax, predicted_class, confidence, true_label):
        """Plot prediction information."""
        ax.text(0.1, 0.8, f'Predicted Class: {predicted_class}', fontsize=14, fontweight='bold')
        ax.text(0.1, 0.6, f'Confidence: {confidence:.4f}', fontsize=12)
        if true_label is not None:
            ax.text(0.1, 0.4, f'True Label: {true_label}', fontsize=12)
            correct = "✓" if predicted_class == true_label else "✗"
            ax.text(0.1, 0.2, f'Correct: {correct}', fontsize=12, 
                   color='green' if correct == "✓" else 'red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _plot_activation_overview(self, ax, activation, layer_name):
        """Plot overview statistics of activations."""
        if activation.dim() == 4:  # Conv layer (B, C, H, W)
            act = activation.squeeze(0)  # Remove batch dim
            
            # Calculate statistics
            mean_activation = act.mean(dim=(1, 2))  # Mean across spatial dims
            max_activation = act.max(dim=2)[0].max(dim=1)[0]  # Max across spatial dims
            
            n_channels = len(mean_activation)
            channels = range(n_channels)
            
            # Create bar plot
            bar1 = ax.bar(channels, mean_activation.numpy(), alpha=0.7, label='Mean', width=0.8)
            bar2 = ax.bar(channels, max_activation.numpy(), alpha=0.5, label='Max', width=0.8)
            
            # Set title and labels with proper spacing
            ax.set_title(f'{layer_name}\nActivation Statistics', fontsize=10, pad=15)
            ax.set_xlabel('Channel', fontsize=9)
            ax.set_ylabel('Activation', fontsize=9)
            
            # Adjust x-axis ticks to prevent crowding
            if n_channels > 20:
                # Show every nth tick to prevent overlap
                step = max(1, n_channels // 10)
                ax.set_xticks(range(0, n_channels, step))
                ax.set_xticklabels(range(0, n_channels, step), fontsize=8)
            else:
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
            
            # Position legend to avoid overlap
            ax.legend(loc='upper right', fontsize=8)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
            
        else:
            ax.text(0.5, 0.5, f'{layer_name}\nNon-conv layer', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
    
    def _plot_feature_maps(self, ax, activation, layer_name):
        """Plot individual feature maps."""
        if activation.dim() == 4:  # Conv layer (B, C, H, W)
            act = activation.squeeze(0)  # Remove batch dim (C, H, W)
            
            # Show ALL channels, not just top 16
            n_channels = act.shape[0]
            
            # Determine optimal grid layout based on number of channels
            if n_channels <= 16:
                cols = min(8, n_channels)
            elif n_channels <= 64:
                cols = 8
            elif n_channels <= 256:
                cols = 16
            else:
                cols = 20  # For very deep layers
            
            rows = (n_channels + cols - 1) // cols
            
            # Calculate spacing to prevent overlap
            margin = 0.01  # Small margin between subplots
            subplot_width = (1.0 - (cols + 1) * margin) / cols
            subplot_height = (1.0 - (rows + 1) * margin) / rows
            
            # Adjust font size based on number of channels
            if n_channels <= 16:
                font_size = 8
            elif n_channels <= 64:
                font_size = 6
            else:
                font_size = 4
            
            for i in range(n_channels):
                feature_map = act[i].numpy()
                
                # Create subplot position
                row = i // cols
                col = i % cols
                
                # Calculate position with margins to prevent overlap
                x_start = margin + col * (subplot_width + margin)
                y_start = 1.0 - margin - (row + 1) * (subplot_height + margin)
                
                # Create inset axis with calculated dimensions
                inset_ax = ax.inset_axes([x_start, y_start, subplot_width, subplot_height])
                
                # Plot feature map
                im = inset_ax.imshow(feature_map, cmap='viridis', aspect='auto')
                
                # Add title with channel number, positioned above the subplot
                if font_size >= 6:
                    inset_ax.set_title(f'{i}', fontsize=font_size, pad=1)
                
                inset_ax.axis('off')
            
            # Set main title with channel count
            ax.set_title(f'{layer_name} - All {n_channels} Feature Maps', fontsize=12, pad=20)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'{layer_name}\nNon-conv layer', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    def _plot_classifier_activations(self, ax, classifier_output):
        """Plot classifier layer activations."""
        output = classifier_output.squeeze(0).numpy()  # Remove batch dim
        
        # Plot top predictions
        top_indices = np.argsort(output)[-20:]  # Top 20
        top_values = output[top_indices]
        
        bars = ax.bar(range(len(top_indices)), top_values, width=0.8)
        ax.set_title('Top 20 Classifier Outputs', fontsize=14, pad=20)
        ax.set_xlabel('Class Index', fontsize=12)
        ax.set_ylabel('Output Value', fontsize=12)
        
        # Set x-axis ticks and labels with proper spacing
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(top_indices, rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Color the highest bars differently (top 3)
        if len(bars) >= 3:
            bars[-1].set_color('red')      # Highest
            bars[-2].set_color('orange')   # Second highest
            bars[-3].set_color('yellow')   # Third highest
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout to prevent label cutoff
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.margins(x=0.01)
    
    def cleanup(self):
        """Remove hooks to free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    """Main function to run activation visualization."""
    print("🎨 CNN Activation Visualizer")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("📦 Loading model...")
    model = CNN(use_prelu=False, use_builtin_conv=True, num_classes=1000)
    
    # Try to load trained weights if available
    results_dir = Path("./results/cnn_results_relu")
    if results_dir.exists():
        # Look for checkpoint files
        checkpoint_files = list(results_dir.glob("**/pytorch_model.bin"))
        if checkpoint_files:
            print(f"📥 Loading trained weights from {checkpoint_files[0]}")
            try:
                state_dict = torch.load(checkpoint_files[0], map_location=device)
                model.load_state_dict(state_dict)
                print("✅ Loaded trained weights")
            except Exception as e:
                print(f"⚠️  Could not load weights: {e}")
                print("Using random weights instead")
        else:
            print("⚠️  No trained weights found, using random weights")
    else:
        print("⚠️  No results directory found, using random weights")
    
    model = model.to(device)
    
    # Load dataset
    print("📊 Loading dataset...")
    try:
        dataset_path = "./processed_datasets/imagenet_processor"
        dataset = load_from_disk(dataset_path)
        eval_dataset = dataset["validation"]
        print(f"✅ Loaded dataset with {len(eval_dataset)} validation samples")
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        print("Please ensure the processed dataset exists at ./processed_datasets/imagenet_processor")
        return
    
    # Select a sample to visualize
    sample_idx = 0  # You can change this to visualize different samples
    sample = eval_dataset[sample_idx]
    
    print(f"🔍 Visualizing sample {sample_idx}")
    print(f"   Label: {sample['labels']}")
    
    # Convert sample to tensor
    pixel_values = torch.tensor(sample['pixel_values'], dtype=torch.float32)
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension
    
    # Create visualizer
    visualizer = ActivationVisualizer(model, device)
    
    # Create output directory
    output_dir = Path("./visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize activations
    save_path = output_dir / f"activations_sample_{sample_idx}.png"
    visualizer.visualize_sample(
        pixel_values, 
        label=sample['labels'], 
        save_path=save_path
    )
    
    # Cleanup
    visualizer.cleanup()
    
    print("✅ Visualization completed!")


if __name__ == "__main__":
    main() 