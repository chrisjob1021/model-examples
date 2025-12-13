"""
Waymo Open Motion Dataset loader for trajectory prediction.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path
import pickle
from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")
    tf = None


class WaymoMotionDataset(Dataset):
    """
    Dataset class for Waymo Open Motion Dataset.
    
    The dataset contains vehicle and pedestrian trajectories with:
    - 1 second of history (10 timesteps at 10Hz)
    - 8 seconds of future to predict (80 timesteps at 10Hz)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'training',
        hist_len: int = 10,  # 1 second at 10Hz
        pred_len: int = 80,  # 8 seconds at 10Hz
        cache_path: Optional[str] = None,
        max_files: Optional[int] = None,
        max_neighbors: int = 20,
        neighbor_radius: float = 50.0
    ):
        """
        Initialize Waymo Motion Dataset.
        
        Args:
            data_path: Path to Waymo dataset directory
            split: Dataset split ('training', 'validation', 'testing')
            hist_len: Number of history timesteps
            pred_len: Number of prediction timesteps
            cache_path: Path to cache processed data
            max_files: Maximum number of TFRecord files to load (for debugging)
            max_neighbors: Maximum number of neighboring agents to consider
            neighbor_radius: Radius in meters to consider neighbors
        """
        self.data_path = Path(data_path)
        self.split = split
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_neighbors = max_neighbors
        self.neighbor_radius = neighbor_radius
        
        # Check if TensorFlow is available
        if tf is None:
            print("Using synthetic data as TensorFlow is not installed")
            self.trajectories = self._generate_synthetic_data()
        else:
            # Load or process data
            self.trajectories = self._load_data(max_files)
        
        print(f"Loaded {len(self.trajectories)} trajectory samples from {split} set")
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic trajectory data for testing without TensorFlow."""
        print(f"Generating synthetic data for {self.split}...")
        
        num_samples = 1000 if self.split == 'training' else 200
        trajectories = []
        
        for i in range(num_samples):
            # Generate random trajectory
            # Start position
            start_x = np.random.uniform(-50, 50)
            start_y = np.random.uniform(-50, 50)
            
            # Velocity
            vx = np.random.uniform(-10, 10)  # m/s
            vy = np.random.uniform(-10, 10)  # m/s
            
            # Generate trajectory points
            total_len = self.hist_len + self.pred_len
            t = np.arange(total_len) * 0.1  # 10Hz sampling
            
            # Add some acceleration/curvature
            ax = np.random.uniform(-1, 1)
            ay = np.random.uniform(-1, 1)
            
            x = start_x + vx * t + 0.5 * ax * t**2
            y = start_y + vy * t + 0.5 * ay * t**2
            
            trajectory = np.stack([x, y], axis=-1)  # [T, 2]
            
            # Split into history and future
            hist = trajectory[:self.hist_len]
            fut = trajectory[self.hist_len:self.hist_len + self.pred_len]
            
            # Generate random neighbors
            num_neighbors = np.random.randint(0, min(5, self.max_neighbors))
            neighbors = []
            
            for _ in range(num_neighbors):
                # Random neighbor trajectory near the agent
                neighbor_start_x = start_x + np.random.uniform(-20, 20)
                neighbor_start_y = start_y + np.random.uniform(-20, 20)
                neighbor_vx = np.random.uniform(-10, 10)
                neighbor_vy = np.random.uniform(-10, 10)
                
                neighbor_x = neighbor_start_x + neighbor_vx * t[:self.hist_len]
                neighbor_y = neighbor_start_y + neighbor_vy * t[:self.hist_len]
                neighbor_traj = np.stack([neighbor_x, neighbor_y], axis=-1)
                neighbors.append(neighbor_traj)
            
            trajectories.append({
                'hist': hist.astype(np.float32),
                'fut': fut.astype(np.float32),
                'neighbors': neighbors
            })
        
        return trajectories
    
    def _load_data(self, max_files: Optional[int] = None) -> List[Dict]:
        """Load and process Waymo TFRecord files."""
        
        # Check cache first
        if self.cache_path:
            cache_file = self.cache_path / f"waymo_{self.split}_{self.hist_len}_{self.pred_len}.pkl"
            if cache_file.exists():
                print(f"Loading cached data from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # Find TFRecord files
        tfrecord_pattern = str(self.data_path / self.split / "*.tfrecord")
        tfrecord_files = sorted(glob.glob(tfrecord_pattern))
        
        if not tfrecord_files:
            print(f"No TFRecord files found at {tfrecord_pattern}")
            print("Generating synthetic data instead...")
            return self._generate_synthetic_data()
        
        if max_files:
            tfrecord_files = tfrecord_files[:max_files]
        
        print(f"Processing {len(tfrecord_files)} TFRecord files...")
        
        trajectories = []
        
        # Process each TFRecord file
        for file_path in tqdm(tfrecord_files, desc="Loading TFRecords"):
            file_trajectories = self._process_tfrecord(file_path)
            trajectories.extend(file_trajectories)
        
        # Cache processed data
        if self.cache_path and trajectories:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_path / f"waymo_{self.split}_{self.hist_len}_{self.pred_len}.pkl"
            print(f"Caching processed data to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(trajectories, f)
        
        return trajectories
    
    def _process_tfrecord(self, file_path: str) -> List[Dict]:
        """Process a single TFRecord file."""
        trajectories = []
        
        try:
            # Parse TFRecord
            dataset = tf.data.TFRecordDataset(file_path)
            
            for raw_record in dataset.take(100):  # Limit samples per file for now
                # Parse the record
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                # Extract trajectory data
                # Note: This is a simplified extraction - actual Waymo format is more complex
                features = example.features.feature
                
                # Try to extract state features (x, y positions)
                if 'state/x' in features and 'state/y' in features:
                    x_values = features['state/x'].float_list.value
                    y_values = features['state/y'].float_list.value
                    
                    if len(x_values) >= self.hist_len + self.pred_len:
                        # Create trajectory
                        positions = np.stack([x_values, y_values], axis=-1)
                        
                        hist = positions[:self.hist_len]
                        fut = positions[self.hist_len:self.hist_len + self.pred_len]
                        
                        # For now, no neighbors from actual data
                        trajectories.append({
                            'hist': hist.astype(np.float32),
                            'fut': fut.astype(np.float32),
                            'neighbors': []
                        })
                
                # If we have enough samples, stop
                if len(trajectories) >= 10:
                    break
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # If we can't process the file, generate some synthetic data
            if not trajectories:
                return self._generate_synthetic_data()[:10]
        
        return trajectories
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single trajectory sample."""
        sample = self.trajectories[idx]
        
        # Convert to tensors
        hist = torch.from_numpy(sample['hist'])
        fut = torch.from_numpy(sample['fut'])
        
        # Handle neighbors
        neighbors = sample['neighbors']
        if not neighbors:
            # Create empty neighbor tensor
            neighbors = torch.zeros((0, self.hist_len, 2))
        else:
            # Stack neighbor trajectories
            neighbors = torch.stack([torch.from_numpy(n) for n in neighbors])
        
        return {
            'hist': hist,
            'fut': fut,
            'neighbors': neighbors
        }


def get_waymo_dataloaders(
    data_path: str,
    batch_size: int = 32,
    hist_len: int = 10,
    pred_len: int = 80,
    cache_path: Optional[str] = None,
    num_workers: int = 4,
    max_files: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for Waymo Motion Dataset.
    
    Args:
        data_path: Path to Waymo dataset directory
        batch_size: Batch size for DataLoader
        hist_len: Number of history timesteps
        pred_len: Number of prediction timesteps
        cache_path: Path to cache processed data
        num_workers: Number of workers for DataLoader
        max_files: Maximum number of TFRecord files to load
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = WaymoMotionDataset(
        data_path=data_path,
        split='training',
        hist_len=hist_len,
        pred_len=pred_len,
        cache_path=cache_path,
        max_files=max_files
    )
    
    val_dataset = WaymoMotionDataset(
        data_path=data_path,
        split='validation',
        hist_len=hist_len,
        pred_len=pred_len,
        cache_path=cache_path,
        max_files=max_files if max_files else 10  # Use fewer files for validation
    )
    
    # Custom collate function to handle variable number of neighbors
    def collate_fn(batch):
        hist = torch.stack([item['hist'] for item in batch])
        fut = torch.stack([item['fut'] for item in batch])
        neighbors = [item['neighbors'] for item in batch]
        
        return {
            'hist': hist,
            'fut': fut,
            'neighbors': neighbors
        }
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader