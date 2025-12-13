import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from .encoder_decoder import EncoderDecoder
from .social_pooling import ConvolutionalSocialPooling, SpatialGrid


class TrajectoryPredictionModel(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 2,
                 encoder_dim: int = 128, decoder_dim: int = 128,
                 num_layers: int = 1, num_modes: int = 6,
                 grid_size: Tuple[int, int] = (13, 3),
                 soc_conv_depth: int = 64, conv_3x1_depth: int = 16,
                 conv_1x1_depth: int = 32):
        super(TrajectoryPredictionModel, self).__init__()
        
        self.encoder_decoder = EncoderDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=encoder_dim,
            num_layers=num_layers,
            num_modes=num_modes,
            social_pooling_dim=soc_conv_depth
        )
        
        self.social_pooling = ConvolutionalSocialPooling(
            encoder_dim=encoder_dim,
            grid_size=grid_size,
            soc_conv_depth=soc_conv_depth,
            conv_3x1_depth=conv_3x1_depth,
            conv_1x1_depth=conv_1x1_depth
        )
        
        self.spatial_grid = SpatialGrid(grid_size=grid_size)
        
        self.encoder_dim = encoder_dim
        self.num_modes = num_modes
        
    def forward(self, hist: torch.Tensor, neighbors_data: List[List[Dict]],
                pred_len: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = hist.size(0)
        device = hist.device
        
        encoder_outputs, hidden_states = self.encoder_decoder.encoder(hist)
        
        all_positions = []
        all_indices = []
        batch_mapping = []
        
        for batch_idx in range(batch_size):
            ego_last_pos = hist[batch_idx, -1, :]
            all_positions.append(torch.zeros(2).to(device)) 
            
            neighbors = neighbors_data[batch_idx]
            neighbor_indices = []
            
            for neighbor in neighbors:
                rel_pos = torch.tensor(neighbor['relative_pos'], dtype=torch.float32).to(device)
                all_positions.append(rel_pos)
                neighbor_indices.append(len(all_positions) - 1)
            
            all_indices.append(neighbor_indices)
            batch_mapping.append(batch_idx)
        
        if len(all_positions) > 0:
            all_positions_tensor = torch.stack(all_positions)
            grid_positions = self.spatial_grid.get_grid_position(all_positions_tensor)
        else:
            grid_positions = torch.zeros(0, 2).to(device)
        
        expanded_encoder_outputs = []
        for batch_idx in range(batch_size):
            expanded_encoder_outputs.append(encoder_outputs[batch_idx])
            
            for _ in all_indices[batch_idx]:
                expanded_encoder_outputs.append(encoder_outputs[batch_idx])
        
        if len(expanded_encoder_outputs) > batch_size:
            expanded_encoder_outputs = torch.stack(expanded_encoder_outputs)
        else:
            expanded_encoder_outputs = encoder_outputs
        
        social_features = self.social_pooling(
            expanded_encoder_outputs[:len(all_positions)],
            all_indices,
            grid_positions
        )
        
        predictions, mode_probs = self.encoder_decoder.decoder(
            social_features,
            hidden_states,
            pred_len
        )
        
        return predictions, mode_probs


class MultiModalLoss(nn.Module):
    def __init__(self, num_modes: int = 6, regression_loss_weight: float = 5.0):
        super(MultiModalLoss, self).__init__()
        self.num_modes = num_modes
        self.regression_loss_weight = regression_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions: torch.Tensor, mode_probs: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        batch_size, pred_len, num_modes, _ = predictions.shape
        
        target_expanded = target.unsqueeze(2).expand(-1, -1, num_modes, -1)
        
        distances = torch.norm(predictions - target_expanded, dim=-1)
        
        mean_distances = distances.mean(dim=1)
        
        best_mode = torch.argmin(mean_distances, dim=-1)
        
        classification_loss = self.ce_loss(
            mode_probs.view(batch_size * pred_len, num_modes),
            best_mode.unsqueeze(1).expand(-1, pred_len).contiguous().view(-1)
        )
        
        regression_losses = []
        for i in range(batch_size):
            mode_predictions = predictions[i, :, best_mode[i], :]
            regression_loss = nn.functional.mse_loss(mode_predictions, target[i])
            regression_losses.append(regression_loss)
        
        regression_loss = torch.stack(regression_losses).mean()
        
        total_loss = classification_loss + self.regression_loss_weight * regression_loss
        
        min_ade = torch.min(mean_distances, dim=-1)[0].mean()
        min_fde = torch.min(distances[:, -1, :], dim=-1)[0].mean()
        
        metrics = {
            'classification_loss': classification_loss.item(),
            'regression_loss': regression_loss.item(),
            'min_ade': min_ade.item(),
            'min_fde': min_fde.item()
        }
        
        return total_loss, metrics