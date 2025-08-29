#!/usr/bin/env python3
"""
Setup script for Waymo Open Motion Dataset.

This script helps download and prepare the Waymo Open Motion Dataset for training.
The dataset requires registration and acceptance of terms at https://waymo.com/open/
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = {
        'tensorflow': 'pip install tensorflow',
        'waymo_open_dataset': 'pip install waymo-open-dataset-tf-2-11-0'
    }
    
    missing = []
    for package, install_cmd in dependencies.items():
        try:
            if package == 'waymo_open_dataset':
                import waymo_open_dataset
            else:
                __import__(package)
        except ImportError:
            missing.append((package, install_cmd))
    
    if missing:
        print("Missing dependencies:")
        for package, cmd in missing:
            print(f"  - {package}: Install with '{cmd}'")
        print("\nNote: waymo-open-dataset requires specific TensorFlow versions.")
        print("Check compatibility at: https://github.com/waymo-research/waymo-open-dataset")
        return False
    
    return True


def download_waymo_dataset(output_dir: Path):
    """
    Guide user through Waymo dataset download process.
    """
    print("=" * 80)
    print("WAYMO OPEN MOTION DATASET SETUP")
    print("=" * 80)
    print()
    print("The Waymo Open Motion Dataset contains:")
    print("  - 103,354 segments of 20 seconds each")
    print("  - 9.9 million unique agent trajectories")
    print("  - 1,750 square km of mapped area")
    print()
    print("To download the dataset:")
    print()
    print("1. Register at: https://waymo.com/open/")
    print("2. Accept the license terms")
    print("3. Download the Motion Dataset v1.3.0")
    print("4. Navigate to: uncompressed/tf_example/")
    print()
    print("Download options:")
    print("  - Full dataset: ~1TB (all splits)")
    print("  - Training only: ~700GB")
    print("  - Validation only: ~150GB")
    print("  - Testing only: ~150GB")
    print()
    print("Recommended structure:")
    print(f"  {output_dir}/")
    print(f"    ├── training/")
    print(f"    │   └── *.tfrecord files")
    print(f"    ├── validation/")
    print(f"    │   └── *.tfrecord files")
    print(f"    └── testing/")
    print(f"        └── *.tfrecord files")
    print()
    print("You can use gsutil to download:")
    print("  gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/* ./training/")
    print("  gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/validation/* ./validation/")
    print("  gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/testing/* ./testing/")
    print()
    print("=" * 80)


def create_sample_data(output_dir: Path):
    """Create sample data for testing without downloading the full dataset."""
    print("\nCreating sample data for testing...")
    
    # Create directory structure
    for split in ['training', 'validation', 'testing']:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a marker file
        marker_file = split_dir / 'SAMPLE_DATA.txt'
        marker_file.write_text(
            "This is sample data for testing.\n"
            "To use real Waymo data, download the dataset and place TFRecord files here.\n"
        )
    
    print(f"Sample data structure created at: {output_dir}")
    print("The model will generate synthetic data when real Waymo data is not found.")


def verify_dataset(data_dir: Path):
    """Verify that the dataset is properly set up."""
    print("\nVerifying dataset structure...")
    
    found_splits = []
    for split in ['training', 'validation', 'testing']:
        split_dir = data_dir / split
        if split_dir.exists():
            tfrecords = list(split_dir.glob("*.tfrecord"))
            if tfrecords:
                found_splits.append(f"  ✓ {split}: {len(tfrecords)} TFRecord files")
            else:
                found_splits.append(f"  ✗ {split}: directory exists but no TFRecord files found")
        else:
            found_splits.append(f"  ✗ {split}: directory not found")
    
    print("Dataset status:")
    for status in found_splits:
        print(status)
    
    if all("✓" in s for s in found_splits):
        print("\n✅ Dataset is properly set up!")
        return True
    else:
        print("\n⚠️  Dataset is not fully set up. The model will use synthetic data for missing splits.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Setup Waymo Open Motion Dataset')
    parser.add_argument('--data_dir', type=str, default='selfdriving/data/waymo',
                       help='Directory to store Waymo dataset')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check if dataset is set up correctly')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample data structure for testing')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.check_only:
        if data_dir.exists():
            verify_dataset(data_dir)
        else:
            print(f"Data directory does not exist: {data_dir}")
            sys.exit(1)
    else:
        print("Checking dependencies...")
        deps_ok = check_dependencies()
        if not deps_ok:
            print("\n⚠️  Some dependencies are missing but the model can still work with synthetic data.")
        
        # Create data directory
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Show download instructions
        download_waymo_dataset(data_dir)
        
        if args.create_sample:
            create_sample_data(data_dir)
        
        # Verify setup
        verify_dataset(data_dir)
        
        print("\nTo train with Waymo dataset:")
        print(f"  python selfdriving/train_hf.py --dataset_name waymo --data_path {data_dir}")


if __name__ == '__main__':
    main()