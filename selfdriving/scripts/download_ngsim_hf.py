#!/usr/bin/env python3
"""
Download and prepare NGSIM dataset using Hugging Face datasets library.
"""
import os
import requests
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
import numpy as np

def download_ngsim_data():
    """Download NGSIM dataset from official sources."""
    
    # NGSIM US-101 dataset URLs
    urls = {
        'us101_trajectories': 'https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD',
        'i80_trajectories': 'https://data.transportation.gov/api/views/4qbx-egtn/rows.csv?accessType=DOWNLOAD'
    }
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {}
    
    for name, url in urls.items():
        file_path = data_dir / f'{name}.csv'
        
        if not file_path.exists():
            print(f"Downloading {name} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {name} to {file_path}")
        else:
            print(f"{name} already exists at {file_path}")
        
        # Load and process the data
        df = pd.read_csv(file_path)
        datasets[name] = df
    
    return datasets

def create_hf_dataset(df, name='ngsim'):
    """Convert NGSIM dataframe to Hugging Face dataset format."""
    
    # Rename columns to standard names if needed
    column_mapping = {
        'Vehicle_ID': 'vehicle_id',
        'Frame_ID': 'frame_id',
        'Total_Frames': 'total_frames',
        'Global_Time': 'global_time',
        'Local_X': 'x',
        'Local_Y': 'y',
        'Global_X': 'global_x',
        'Global_Y': 'global_y',
        'v_Length': 'length',
        'v_Width': 'width',
        'v_Class': 'vehicle_class',
        'v_Vel': 'velocity',
        'v_Acc': 'acceleration',
        'Lane_ID': 'lane_id',
        'Preceding': 'preceding',
        'Following': 'following',
        'Space_Headway': 'space_headway',
        'Time_Headway': 'time_headway'
    }
    
    # Rename columns if they exist
    df_renamed = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Create sequences for each vehicle
    sequences = []
    
    for vehicle_id in df_renamed['vehicle_id'].unique():
        vehicle_data = df_renamed[df_renamed['vehicle_id'] == vehicle_id].sort_values('frame_id')
        
        if len(vehicle_data) >= 80:  # At least 8 seconds at 10Hz
            sequences.append({
                'vehicle_id': int(vehicle_id),
                'trajectory': vehicle_data[['x', 'y']].values.tolist(),
                'velocity': vehicle_data['velocity'].values.tolist(),
                'acceleration': vehicle_data['acceleration'].values.tolist(),
                'frame_ids': vehicle_data['frame_id'].values.tolist(),
                'lane_ids': vehicle_data['lane_id'].values.tolist(),
                'length': len(vehicle_data)
            })
    
    # Create HF dataset
    dataset = Dataset.from_list(sequences)
    
    # Split into train/val/test
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })
    
    return dataset_dict

def main():
    print("Downloading NGSIM dataset...")
    datasets = download_ngsim_data()
    
    # Process US-101 dataset
    if 'us101_trajectories' in datasets:
        print("\nProcessing US-101 dataset...")
        us101_df = datasets['us101_trajectories']
        print(f"US-101 shape: {us101_df.shape}")
        print(f"Columns: {us101_df.columns.tolist()}")
        
        hf_dataset = create_hf_dataset(us101_df, 'ngsim_us101')
        
        # Save to disk
        hf_dataset.save_to_disk('data/processed/ngsim_us101')
        print(f"Saved HF dataset to data/processed/ngsim_us101")
        
        # Print dataset info
        print(f"\nDataset splits:")
        for split, data in hf_dataset.items():
            print(f"  {split}: {len(data)} sequences")
    
    # Process I-80 dataset  
    if 'i80_trajectories' in datasets:
        print("\nProcessing I-80 dataset...")
        i80_df = datasets['i80_trajectories']
        print(f"I-80 shape: {i80_df.shape}")
        
        hf_dataset = create_hf_dataset(i80_df, 'ngsim_i80')
        hf_dataset.save_to_disk('data/processed/ngsim_i80')
        print(f"Saved HF dataset to data/processed/ngsim_i80")

if __name__ == "__main__":
    main()