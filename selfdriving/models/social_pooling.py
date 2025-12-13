import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class ConvolutionalSocialPooling(nn.Module):
    def __init__(self, encoder_dim: int = 128, grid_size: Tuple[int, int] = (13, 3),
                 soc_conv_depth: int = 64, conv_3x1_depth: int = 16,
                 conv_1x1_depth: int = 32, pool_size: Tuple[int, int] = (3, 3)):
        super(ConvolutionalSocialPooling, self).__init__()
        
        self.grid_size = grid_size
        self.pool_size = pool_size
        self.encoder_dim = encoder_dim
        
        self.conv_3x1 = nn.Conv2d(encoder_dim, conv_3x1_depth, (3, 1), padding=(1, 0))
        
        self.conv_1x1 = nn.Conv2d(conv_3x1_depth, conv_1x1_depth, (1, 1))
        
        self.maxpool = nn.MaxPool2d(pool_size)
        
        pooled_height = (grid_size[0] + pool_size[0] - 1) // pool_size[0]
        pooled_width = (grid_size[1] + pool_size[1] - 1) // pool_size[1]
        self.fc_dim = conv_1x1_depth * pooled_height * pooled_width
        
        self.fc = nn.Linear(self.fc_dim, soc_conv_depth)
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def create_social_tensor(self, encoder_outputs: torch.Tensor, 
                           neighbor_indices: List[List[int]],
                           grid_positions: torch.Tensor) -> torch.Tensor:
        
        batch_size = len(neighbor_indices)
        social_tensor = torch.zeros(
            batch_size, self.encoder_dim, self.grid_size[0], self.grid_size[1]
        ).to(encoder_outputs.device)
        
        for i in range(batch_size):
            neighbors = neighbor_indices[i]
            if len(neighbors) > 0:
                neighbor_features = encoder_outputs[neighbors]  
                neighbor_positions = grid_positions[neighbors]  
                
                for j, (feat, pos) in enumerate(zip(neighbor_features, neighbor_positions)):
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                        social_tensor[i, :, x, y] = feat[:, -1]  
        
        return social_tensor
    
    def forward(self, encoder_outputs: torch.Tensor,
                neighbor_indices: List[List[int]],
                grid_positions: torch.Tensor) -> torch.Tensor:
        
        social_tensor = self.create_social_tensor(encoder_outputs, neighbor_indices, grid_positions)
        
        x = self.leaky_relu(self.conv_3x1(social_tensor))
        
        x = self.leaky_relu(self.conv_1x1(x))
        
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        
        social_features = self.leaky_relu(self.fc(x))
        
        return social_features


class SpatialGrid:
    def __init__(self, grid_size: Tuple[int, int] = (13, 3),
                 grid_extent: Tuple[float, float] = (90.0, 21.0)):
        
        self.grid_size = grid_size
        self.grid_extent = grid_extent
        self.cell_size = (grid_extent[0] / grid_size[0], grid_extent[1] / grid_size[1])
        
    def get_grid_position(self, relative_position: torch.Tensor) -> torch.Tensor:
        
        grid_x = torch.clamp(
            (relative_position[:, 0] + self.grid_extent[0] / 2) / self.cell_size[0],
            0, self.grid_size[0] - 1
        ).long()
        
        grid_y = torch.clamp(
            (relative_position[:, 1] + self.grid_extent[1] / 2) / self.cell_size[1],
            0, self.grid_size[1] - 1
        ).long()
        
        return torch.stack([grid_x, grid_y], dim=1)
    
    def get_neighbor_indices(self, positions: torch.Tensor, 
                           current_idx: int,
                           max_neighbors: int = 50) -> List[int]:
        
        current_pos = positions[current_idx]
        
        distances = torch.norm(positions - current_pos, dim=1)
        
        neighbor_mask = (distances > 0) & (distances < self.grid_extent[0])
        neighbor_indices = torch.where(neighbor_mask)[0].tolist()
        
        if len(neighbor_indices) > max_neighbors:
            distances_neighbors = distances[neighbor_mask]
            _, sorted_indices = torch.sort(distances_neighbors)
            neighbor_indices = [neighbor_indices[i] for i in sorted_indices[:max_neighbors]]
        
        return neighbor_indices