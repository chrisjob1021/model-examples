import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvolutionalSocialPooling(nn.Module):
    """Convolutional Social Pooling layer for capturing vehicle interactions."""
    
    def __init__(self, input_channels=5, hidden_channels=[64, 128], kernel_size=3):
        """
        Initialize the social pooling layers.
        
        Args:
            input_channels: Number of features per grid cell (5: occupancy, x, y, vx, vy)
            hidden_channels: List of channel sizes for conv layers
            kernel_size: Kernel size for convolutions
        """
        super(ConvolutionalSocialPooling, self).__init__()
        
        # Store parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        # Create each convolutional layer
        for out_channels in hidden_channels:
            # Add conv layer with padding to maintain spatial dimensions
            layers.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size//2  # Same padding
            ))
            # Add batch normalization for stable training
            layers.append(nn.BatchNorm2d(out_channels))
            # Add ReLU activation
            layers.append(nn.ReLU(inplace=True))
            # Update input channels for next layer
            in_channels = out_channels
        
        # Create sequential model from layers
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, social_tensor):
        """
        Process social tensor through convolutional layers.
        
        Args:
            social_tensor: [batch, height, width, channels] tensor
            
        Returns:
            Social features: [batch, final_channels] tensor
        """
        # Reshape from [B, H, W, C] to [B, C, H, W] for Conv2d
        batch_size = social_tensor.size(0)
        social_tensor = social_tensor.permute(0, 3, 1, 2)
        
        # Apply convolutional layers
        features = self.conv_layers(social_tensor)
        
        # Global pooling to get [batch, channels, 1, 1]
        pooled = self.global_pool(features)
        
        # Flatten to [batch, channels]
        output = pooled.view(batch_size, -1)
        
        return output


class TrajectoryEncoder(nn.Module):
    """LSTM encoder for processing historical trajectories."""
    
    def __init__(self, input_dim=5, hidden_dim=256, num_layers=2, dropout=0.1):
        """
        Initialize the trajectory encoder.
        
        Args:
            input_dim: Input feature dimension (x, y, vx, vy, heading)
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(TrajectoryEncoder, self).__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection layer to match LSTM input size
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input shape: [batch, seq, features]
            dropout=dropout if num_layers > 1 else 0  # Only use dropout if multiple layers
        )
        
        # Output projection with dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, history):
        """
        Encode historical trajectory.
        
        Args:
            history: [batch, seq_len, input_dim] tensor
            
        Returns:
            hidden: Final hidden state [num_layers, batch, hidden_dim]
            cell: Final cell state [num_layers, batch, hidden_dim]
        """
        # Project input features to hidden dimension
        x = self.input_proj(history)  # [batch, seq_len, hidden_dim]
        
        # Process sequence through LSTM
        output, (hidden, cell) = self.lstm(x)
        # output: [batch, seq_len, hidden_dim]
        # hidden: [num_layers, batch, hidden_dim]
        # cell: [num_layers, batch, hidden_dim]
        
        return hidden, cell


class TrajectoryDecoder(nn.Module):
    """LSTM decoder for generating future trajectories."""
    
    def __init__(self, output_dim=2, hidden_dim=256, num_layers=2, 
                 num_modes=3, dropout=0.1):
        """
        Initialize the trajectory decoder.
        
        Args:
            output_dim: Output dimension (x, y positions)
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            num_modes: Number of prediction modes (for multi-modal output)
            dropout: Dropout probability
        """
        super(TrajectoryDecoder, self).__init__()
        
        # Store dimensions
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_modes = num_modes
        
        # LSTM for future trajectory generation
        self.lstm = nn.LSTM(
            input_size=output_dim,  # Previous position as input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers for multi-modal predictions
        # Predict positions for each mode
        self.position_proj = nn.Linear(hidden_dim, num_modes * output_dim)
        
        # Predict mode probabilities (which trajectory is most likely)
        self.mode_prob_proj = nn.Linear(hidden_dim, num_modes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, last_position, hidden, cell, future_len):
        """
        Decode future trajectory.
        
        Args:
            last_position: Last observed position [batch, output_dim]
            hidden: Encoder hidden state [num_layers, batch, hidden_dim]
            cell: Encoder cell state [num_layers, batch, hidden_dim]
            future_len: Number of future steps to predict
            
        Returns:
            positions: Predicted positions [batch, future_len, num_modes, output_dim]
            mode_probs: Mode probabilities [batch, num_modes]
        """
        batch_size = last_position.size(0)
        
        # Initialize list to store predictions
        predictions = []
        
        # Use last observed position as first decoder input
        decoder_input = last_position.unsqueeze(1)  # [batch, 1, output_dim]
        
        # Generate future positions autoregressively
        for t in range(future_len):
            # LSTM forward pass
            output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            # output: [batch, 1, hidden_dim]
            
            # Apply dropout
            output = self.dropout(output)
            
            # Predict positions for all modes
            positions = self.position_proj(output)  # [batch, 1, num_modes * output_dim]
            # Reshape to separate modes
            positions = positions.view(batch_size, 1, self.num_modes, self.output_dim)
            
            # Store prediction
            predictions.append(positions)
            
            # Use predicted position from first mode as next input (teacher forcing during training)
            decoder_input = positions[:, :, 0, :]  # [batch, 1, output_dim]
        
        # Concatenate all predictions
        all_positions = torch.cat(predictions, dim=1)  # [batch, future_len, num_modes, output_dim]
        
        # Predict mode probabilities using final hidden state
        final_hidden = hidden[-1]  # [batch, hidden_dim]
        mode_logits = self.mode_prob_proj(final_hidden)  # [batch, num_modes]
        mode_probs = F.softmax(mode_logits, dim=-1)  # Normalize to probabilities
        
        return all_positions, mode_probs


class ConvSocialPoolingModel(nn.Module):
    """Complete model combining encoder, decoder, and social pooling."""
    
    def __init__(self, config):
        """
        Initialize the complete model.
        
        Args:
            config: Configuration dictionary
        """
        super(ConvSocialPoolingModel, self).__init__()
        
        # Extract model configuration
        model_config = config['model']
        data_config = config['data']
        
        # Store important dimensions
        self.hidden_dim = model_config['hidden_dim']
        self.num_modes = model_config['num_modes']
        self.future_len = int(data_config['future_seconds'] * data_config['sample_rate'])
        
        # Initialize encoder for trajectory history
        self.encoder = TrajectoryEncoder(
            input_dim=model_config['input_dim'] + 1,  # +1 for heading
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
        
        # Initialize social pooling for neighbor interactions
        self.social_pooling = ConvolutionalSocialPooling(
            input_channels=5,  # Occupancy + 4 features
            hidden_channels=model_config['social_conv_channels'],
            kernel_size=model_config['social_kernel_size']
        )
        
        # Fusion layer to combine trajectory and social features
        social_feature_dim = model_config['social_conv_channels'][-1]
        self.fusion = nn.Sequential(
            nn.Linear(model_config['hidden_dim'] + social_feature_dim, model_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(model_config['dropout'])
        )
        
        # Initialize decoder for future trajectory
        self.decoder = TrajectoryDecoder(
            output_dim=model_config['output_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_modes=model_config['num_modes'],
            dropout=model_config['dropout']
        )
        
    def forward(self, history, social_tensor):
        """
        Forward pass through the model.
        
        Args:
            history: Historical trajectory [batch, history_len, features]
            social_tensor: Social context grid [batch, height, width, channels]
            
        Returns:
            predictions: Future positions [batch, future_len, num_modes, 2]
            mode_probs: Mode probabilities [batch, num_modes]
        """
        batch_size = history.size(0)
        
        # Encode historical trajectory
        hidden, cell = self.encoder(history)
        # hidden: [num_layers, batch, hidden_dim]
        
        # Extract social features
        social_features = self.social_pooling(social_tensor)
        # social_features: [batch, social_feature_dim]
        
        # Combine trajectory and social information
        # Use top layer hidden state for fusion
        trajectory_features = hidden[-1]  # [batch, hidden_dim]
        combined_features = torch.cat([trajectory_features, social_features], dim=-1)
        # combined_features: [batch, hidden_dim + social_feature_dim]
        
        # Fuse features
        fused_features = self.fusion(combined_features)  # [batch, hidden_dim]
        
        # Update top layer hidden state with fused features
        hidden = hidden.clone()
        hidden[-1] = fused_features
        
        # Get last observed position from history
        last_position = history[:, -1, :2]  # [batch, 2] (x, y only)
        
        # Decode future trajectory
        predictions, mode_probs = self.decoder(
            last_position, hidden, cell, self.future_len
        )
        
        return predictions, mode_probs
    
    def inference(self, history, social_tensor):
        """
        Inference mode - returns best predicted trajectory.
        
        Args:
            history: Historical trajectory [batch, history_len, features]
            social_tensor: Social context grid [batch, height, width, channels]
            
        Returns:
            best_trajectory: Most likely future trajectory [batch, future_len, 2]
            all_trajectories: All predicted modes [batch, future_len, num_modes, 2]
            mode_probs: Mode probabilities [batch, num_modes]
        """
        # Get predictions
        predictions, mode_probs = self.forward(history, social_tensor)
        
        # Select best mode (highest probability)
        best_mode_idx = torch.argmax(mode_probs, dim=-1)  # [batch]
        
        # Extract best trajectory for each sample in batch
        batch_size = predictions.size(0)
        best_trajectory = torch.zeros(batch_size, self.future_len, 2).to(predictions.device)
        
        for i in range(batch_size):
            best_trajectory[i] = predictions[i, :, best_mode_idx[i], :]
        
        return best_trajectory, predictions, mode_probs