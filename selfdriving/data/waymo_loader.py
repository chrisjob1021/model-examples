import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from tqdm import tqdm
import pickle


class WaymoTrajectoryDataset(Dataset):
    """PyTorch Dataset for loading Waymo Open Dataset trajectories."""
    
    def __init__(self, data_path, split='training', config=None, cache_dir='cache'):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to Waymo dataset directory
            split: 'training' or 'testing'
            config: Configuration dictionary
            cache_dir: Directory to store processed cache files
        """
        # Expand ~ to full path and store dataset location
        self.data_path = os.path.expanduser(data_path)
        self.split = split
        self.config = config
        self.cache_dir = cache_dir
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Calculate trajectory lengths based on time and sampling rate
        # e.g., 3 seconds * 10 Hz = 30 frames of history
        self.history_len = int(config['data']['history_seconds'] * config['data']['sample_rate'])
        # e.g., 5 seconds * 10 Hz = 50 frames to predict
        self.future_len = int(config['data']['future_seconds'] * config['data']['sample_rate'])
        # Radius in meters to look for neighboring vehicles
        self.social_radius = config['data']['social_radius']
        # Maximum number of neighbors to consider
        self.max_neighbors = config['data']['max_neighbors']
        
        # Load trajectories from cache or process from scratch
        self.trajectories = self._load_trajectories()
        
    def _load_trajectories(self):
        """Load trajectories from cache or process TFRecord files."""
        # Define cache file path based on data split
        cache_file = os.path.join(self.cache_dir, f'{self.split}_trajectories.pkl')
        
        # Check if we already processed this data
        if os.path.exists(cache_file):
            print(f"Loading cached trajectories from {cache_file}")
            # Load preprocessed trajectories from pickle file
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Processing {self.split} data...")
        trajectories = []
        
        # Find all TFRecord files in the split directory
        pattern = os.path.join(self.data_path, self.split, '*.tfrecord')
        # Get sorted list of files, limit to 10 for testing
        tfrecord_files = sorted(glob.glob(pattern))[:10]  # Start with 10 files for testing
        
        # Process each TFRecord file
        for tfrecord_file in tqdm(tfrecord_files, desc="Processing TFRecords"):
            # Create TensorFlow dataset from TFRecord file
            dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
            
            # Process first 100 frames from each file (for testing)
            for data in dataset.take(100):
                # Parse the frame using Waymo's protobuf format
                frame = dataset_pb2.Frame()
                frame.ParseFromString(data.numpy())
                
                # Extract vehicle trajectory data from this frame
                traj_data = self._extract_trajectories_from_frame(frame)
                if traj_data:
                    # Add extracted trajectories to our list
                    trajectories.extend(traj_data)
        
        # Save processed trajectories to cache for faster loading next time
        with open(cache_file, 'wb') as f:
            pickle.dump(trajectories, f)
        
        print(f"Processed {len(trajectories)} trajectory samples")
        return trajectories
    
    def _extract_trajectories_from_frame(self, frame):
        """Extract vehicle trajectories from a single Waymo frame."""
        trajectories = []
        
        # Dictionary to store all vehicles and their states
        vehicles = {}
        
        # Iterate through all objects in the frame
        for obj in frame.objects:
            # Only process vehicle objects (ignore pedestrians, cyclists, etc.)
            if obj.type == dataset_pb2.Label.TYPE_VEHICLE:
                obj_id = obj.id
                # Initialize list for this vehicle if not seen before
                if obj_id not in vehicles:
                    vehicles[obj_id] = []
                
                # Extract vehicle state information
                state = {
                    'x': obj.box.center_x,  # X position in world coordinates
                    'y': obj.box.center_y,  # Y position in world coordinates
                    'z': obj.box.center_z,  # Z position (height)
                    'length': obj.box.length,  # Vehicle length
                    'width': obj.box.width,  # Vehicle width
                    'height': obj.box.height,  # Vehicle height
                    'heading': obj.box.heading,  # Vehicle heading angle
                    # Velocity components, default to 0 if not available
                    'velocity_x': obj.metadata.speed_x if obj.metadata.speed_x else 0,
                    'velocity_y': obj.metadata.speed_y if obj.metadata.speed_y else 0,
                    'timestamp': frame.timestamp_micros  # Frame timestamp in microseconds
                }
                # Add state to vehicle's trajectory
                vehicles[obj_id].append(state)
        
        # Create training samples from vehicle trajectories
        for vehicle_id, states in vehicles.items():
            # Check if we have enough frames for history + future
            if len(states) >= self.history_len + self.future_len:
                # Create sliding window samples
                for i in range(len(states) - self.history_len - self.future_len + 1):
                    # Extract history window (past trajectory)
                    history = states[i:i + self.history_len]
                    # Extract future window (trajectory to predict)
                    future = states[i + self.history_len:i + self.history_len + self.future_len]
                    
                    # Get current state (last frame of history)
                    curr_state = history[-1]
                    # Find neighboring vehicles at current timestamp
                    neighbors = self._find_neighbors(curr_state, vehicles, vehicle_id)
                    
                    # Create training sample
                    sample = {
                        'history': self._states_to_tensor(history),  # Past trajectory
                        'future': self._states_to_tensor(future),  # Future trajectory to predict
                        'neighbors': neighbors,  # Neighboring vehicles info
                        'vehicle_id': vehicle_id,  # ID of ego vehicle
                        'timestamp': curr_state['timestamp']  # Current timestamp
                    }
                    trajectories.append(sample)
        
        return trajectories
    
    def _states_to_tensor(self, states):
        """Convert list of vehicle states to numpy array."""
        features = []
        # Extract relevant features from each state
        for state in states:
            features.append([
                state['x'],  # X position
                state['y'],  # Y position
                state['velocity_x'],  # X velocity
                state['velocity_y'],  # Y velocity
                state['heading']  # Heading angle
            ])
        # Return as numpy array with shape [seq_len, 5]
        return np.array(features, dtype=np.float32)
    
    def _find_neighbors(self, curr_state, all_vehicles, ego_id):
        """Find neighboring vehicles within social radius."""
        neighbors = []
        # Get ego vehicle position
        ego_x, ego_y = curr_state['x'], curr_state['y']
        
        # Check all other vehicles
        for vehicle_id, states in all_vehicles.items():
            # Skip ego vehicle
            if vehicle_id == ego_id:
                continue
            
            # Find state of this vehicle at the same timestamp
            for state in states:
                # Check if timestamps are close (within 0.1 second)
                if abs(state['timestamp'] - curr_state['timestamp']) < 1e5:  # 100,000 microseconds
                    # Calculate Euclidean distance
                    dist = np.sqrt((state['x'] - ego_x)**2 + (state['y'] - ego_y)**2)
                    # Check if within social radius
                    if dist < self.social_radius:
                        # Store relative position and velocity
                        neighbors.append({
                            'relative_x': state['x'] - ego_x,  # Relative X position
                            'relative_y': state['y'] - ego_y,  # Relative Y position
                            'velocity_x': state['velocity_x'],  # Neighbor's X velocity
                            'velocity_y': state['velocity_y'],  # Neighbor's Y velocity
                            'distance': dist  # Distance to ego vehicle
                        })
                    break  # Found matching timestamp, move to next vehicle
        
        # Sort neighbors by distance (closest first)
        neighbors.sort(key=lambda x: x['distance'])
        # Return only max_neighbors closest vehicles
        return neighbors[:self.max_neighbors]
    
    def __len__(self):
        """Return number of trajectory samples in dataset."""
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """Get a single trajectory sample by index."""
        # Get raw sample from list
        sample = self.trajectories[idx]
        
        # Convert numpy arrays to PyTorch tensors
        history = torch.from_numpy(sample['history'])
        future = torch.from_numpy(sample['future'])
        
        # Create social pooling tensor from neighbors
        social_tensor = self._create_social_tensor(sample['neighbors'])
        
        return {
            'history': history,  # [history_len, 5] tensor
            'future': future,  # [future_len, 5] tensor
            'social_tensor': social_tensor  # [grid_h, grid_w, 5] tensor
        }
    
    def _create_social_tensor(self, neighbors):
        """Create grid-based social tensor for convolutional social pooling."""
        # Get grid dimensions from config
        grid_size = self.config['data']['grid_size']
        # Initialize empty tensor [height, width, features]
        tensor = torch.zeros(grid_size[0], grid_size[1], 5)  # 5 features per cell
        
        # Place each neighbor in the grid
        for neighbor in neighbors:
            # Map relative position to grid cell
            # Convert from [-radius, radius] to [0, grid_size]
            grid_x = int((neighbor['relative_x'] + self.social_radius) / (2 * self.social_radius) * grid_size[0])
            grid_y = int((neighbor['relative_y'] + self.social_radius) / (2 * self.social_radius) * grid_size[1])
            
            # Check if within grid bounds
            if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                tensor[grid_x, grid_y, 0] = 1  # Occupancy flag (1 = occupied)
                tensor[grid_x, grid_y, 1] = neighbor['relative_x']  # Relative X position
                tensor[grid_x, grid_y, 2] = neighbor['relative_y']  # Relative Y position
                tensor[grid_x, grid_y, 3] = neighbor['velocity_x']  # Neighbor X velocity
                tensor[grid_x, grid_y, 4] = neighbor['velocity_y']  # Neighbor Y velocity
        
        return tensor


def create_data_loaders(config):
    """Create PyTorch DataLoaders for training and testing."""
    # Create training dataset
    train_dataset = WaymoTrajectoryDataset(
        config['data']['waymo_path'],  # Path to Waymo data
        split='training',  # Use training split
        config=config  # Pass full config
    )
    
    # Create testing dataset
    test_dataset = WaymoTrajectoryDataset(
        config['data']['waymo_path'],  # Path to Waymo data
        split='testing',  # Use testing split
        config=config  # Pass full config
    )
    
    # Create training DataLoader with shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],  # Batch size from config
        shuffle=config['data']['shuffle'],  # Shuffle training data
        num_workers=config['data']['num_workers'],  # Parallel data loading
        pin_memory=True  # Pin memory for faster GPU transfer
    )
    
    # Create testing DataLoader without shuffling
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],  # Batch size from config
        shuffle=False,  # Don't shuffle test data
        num_workers=config['data']['num_workers'],  # Parallel data loading
        pin_memory=True  # Pin memory for faster GPU transfer
    )
    
    return train_loader, test_loader