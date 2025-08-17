import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd


class NGSIMDataset(Dataset):
    """
    NGSIM Dataset class that wraps HuggingFace datasets functionality
    for trajectory prediction tasks.
    """
    
    def __init__(
        self,
        dataset: HFDataset,
        hist_len: int = 30,
        pred_len: int = 50,
        skip: int = 1,
        grid_size: Tuple[int, int] = (13, 3),
        max_neighbors: int = 50
    ):
        """
        Initialize NGSIM dataset.
        
        Args:
            dataset: HuggingFace dataset object
            hist_len: Length of history trajectory (in frames)
            pred_len: Length of predicted trajectory (in frames)
            skip: Number of frames to skip between samples
            grid_size: Size of spatial grid for social pooling
            max_neighbors: Maximum number of neighbors to consider
        """
        self.dataset = dataset
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.skip = skip
        self.grid_size = grid_size
        self.max_neighbors = max_neighbors
        
        # Process dataset to extract trajectories
        self.trajectories = self._process_dataset()
        
    def _process_dataset(self) -> List[Dict]:
        """
        Process the HuggingFace dataset to extract trajectory samples.
        
        Returns:
            List of trajectory samples
        """
        trajectories = []
        
        # Group by vehicle ID and extract trajectories
        if 'vehicle_id' in self.dataset.column_names:
            # Assuming dataset has columns: vehicle_id, frame, x, y, etc.
            df = self.dataset.to_pandas()
            
            # Group by vehicle
            for vehicle_id, vehicle_data in df.groupby('vehicle_id'):
                vehicle_data = vehicle_data.sort_values('frame')
                
                # Extract windows of hist_len + pred_len
                total_len = self.hist_len + self.pred_len
                
                for i in range(0, len(vehicle_data) - total_len + 1, self.skip):
                    window = vehicle_data.iloc[i:i + total_len]
                    
                    # Extract positions
                    positions = window[['x', 'y']].values
                    hist_positions = positions[:self.hist_len]
                    fut_positions = positions[self.hist_len:]
                    
                    # Extract neighbor information
                    neighbors = self._extract_neighbors(
                        df, 
                        vehicle_id,
                        window['frame'].iloc[self.hist_len - 1],
                        window[['x', 'y']].iloc[self.hist_len - 1].values
                    )
                    
                    trajectories.append({
                        'hist': hist_positions,
                        'fut': fut_positions,
                        'neighbors': neighbors
                    })
        else:
            # Fallback for different dataset format
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                if 'hist' in sample and 'fut' in sample:
                    trajectories.append(sample)
                    
        return trajectories
    
    def _extract_neighbors(
        self,
        df: pd.DataFrame,
        ego_id: int,
        frame: int,
        ego_pos: np.ndarray
    ) -> List[Dict]:
        """
        Extract neighbor vehicles at a given frame.
        
        Args:
            df: Full dataset dataframe
            ego_id: ID of ego vehicle
            frame: Current frame
            ego_pos: Position of ego vehicle
            
        Returns:
            List of neighbor information
        """
        neighbors = []
        
        # Get all vehicles at the current frame
        frame_data = df[df['frame'] == frame]
        
        for _, neighbor in frame_data.iterrows():
            if neighbor['vehicle_id'] != ego_id:
                neighbor_pos = neighbor[['x', 'y']].values
                relative_pos = neighbor_pos - ego_pos
                
                # Only include neighbors within reasonable distance
                distance = np.linalg.norm(relative_pos)
                if distance < 100:  # meters
                    neighbors.append({
                        'vehicle_id': neighbor['vehicle_id'],
                        'relative_pos': relative_pos.tolist(),
                        'distance': distance
                    })
        
        # Sort by distance and limit to max_neighbors
        neighbors = sorted(neighbors, key=lambda x: x['distance'])
        neighbors = neighbors[:self.max_neighbors]
        
        return neighbors
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing trajectory data
        """
        sample = self.trajectories[idx]
        
        return {
            'hist': torch.FloatTensor(sample['hist']),
            'fut': torch.FloatTensor(sample['fut']),
            'neighbors': sample.get('neighbors', [])
        }


def load_ngsim_from_huggingface(
    dataset_name: str = "ngsim",
    split: str = "train",
    **kwargs
) -> HFDataset:
    """
    Load NGSIM dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split to load
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        HuggingFace dataset object
    """
    try:
        # Try to load from HuggingFace
        dataset = load_dataset(dataset_name, split=split, **kwargs)
    except Exception as e:
        print(f"Could not load dataset from HuggingFace: {e}")
        # Fallback to local loading if needed
        dataset = None
        
    return dataset


def get_ngsim_dataloaders(
    data_path: Optional[str] = None,
    dataset_name: str = "ngsim",
    batch_size: int = 128,
    hist_len: int = 30,
    pred_len: int = 50,
    train_split: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get DataLoaders for NGSIM dataset using HuggingFace datasets.
    
    Args:
        data_path: Optional path to local data (will try HuggingFace first)
        dataset_name: Name of dataset on HuggingFace
        batch_size: Batch size for DataLoader
        hist_len: Length of history trajectory
        pred_len: Length of predicted trajectory
        train_split: Fraction of data to use for training
        num_workers: Number of workers for DataLoader
        
    Returns:
        Train and validation DataLoaders
    """
    # Try to load from HuggingFace first
    hf_dataset = load_ngsim_from_huggingface(dataset_name)
    
    if hf_dataset is None and data_path:
        # Fallback to loading from local file
        print(f"Loading dataset from local path: {data_path}")
        # This is a placeholder - implement based on your local data format
        import pandas as pd
        df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_pickle(data_path)
        
        # Convert to HuggingFace dataset format
        from datasets import Dataset as HFDataset
        hf_dataset = HFDataset.from_pandas(df)
    
    if hf_dataset is None:
        raise ValueError("Could not load dataset from HuggingFace or local path")
    
    # Split dataset
    if 'train' in hf_dataset:
        train_data = hf_dataset['train']
        val_data = hf_dataset.get('validation', hf_dataset.get('test'))
    else:
        # Manual split
        split_idx = int(len(hf_dataset) * train_split)
        train_data = hf_dataset.select(range(split_idx))
        val_data = hf_dataset.select(range(split_idx, len(hf_dataset)))
    
    # Create PyTorch datasets
    train_dataset = NGSIMDataset(
        train_data,
        hist_len=hist_len,
        pred_len=pred_len
    )
    
    val_dataset = NGSIMDataset(
        val_data,
        hist_len=hist_len,
        pred_len=pred_len
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader