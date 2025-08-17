import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 1):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim: int = 2, hidden_dim: int = 128, 
                 num_layers: int = 1, num_modes: int = 6):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_modes = num_modes
        
        self.lstm = nn.LSTM(
            input_size=output_dim + hidden_dim,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim * num_modes)
        
        self.mode_prob_layer = nn.Linear(hidden_dim, num_modes)
        
    def forward(self, social_features: torch.Tensor, 
                hidden_state: Tuple[torch.Tensor, torch.Tensor],
                pred_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = social_features.size(0)
        hidden, cell = hidden_state
        
        predictions = []
        mode_probs = []
        
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(social_features.device)
        
        for t in range(pred_len):
            decoder_input_combined = torch.cat([decoder_input, social_features.unsqueeze(1)], dim=-1)
            
            output, (hidden, cell) = self.lstm(decoder_input_combined, (hidden, cell))
            
            pred = self.output_layer(output)
            pred = pred.view(batch_size, self.num_modes, self.output_dim)
            predictions.append(pred)
            
            mode_prob = F.softmax(self.mode_prob_layer(output.squeeze(1)), dim=-1)
            mode_probs.append(mode_prob)
            
            decoder_input = pred[:, 0, :].unsqueeze(1)
        
        predictions = torch.stack(predictions, dim=1)
        mode_probs = torch.stack(mode_probs, dim=1)
        
        return predictions, mode_probs


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, 
                 hidden_dim: int = 128, num_layers: int = 1,
                 num_modes: int = 6, social_pooling_dim: int = 64):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = LSTMDecoder(output_dim, hidden_dim, num_layers, num_modes)
        self.social_pooling_dim = social_pooling_dim
        
    def forward(self, hist: torch.Tensor, social_features: torch.Tensor, 
                pred_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        _, hidden_state = self.encoder(hist)
        
        predictions, mode_probs = self.decoder(social_features, hidden_state, pred_len)
        
        return predictions, mode_probs