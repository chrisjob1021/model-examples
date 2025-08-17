#!/usr/bin/env python3

import os
import urllib.request
import zipfile
import argparse
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    parser = argparse.ArgumentParser(description='Download NGSIM dataset')
    parser.add_argument('--output_dir', type=str, default='selfdriving/data/raw',
                       help='Directory to save downloaded data')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("NGSIM Dataset Download Instructions")
    print("="*50)
    print("The NGSIM dataset must be downloaded from the official source:")
    print("https://www.fhwa.dot.gov/publications/research/operations/07030/")
    print()
    print("Required files:")
    print("1. US-101 Dataset: trajectories-0750am-0805am.txt")
    print("2. I-80 Dataset: trajectories-0400-0415.txt")
    print()
    print("After downloading, place the files in:")
    print(f"  {args.output_dir}")
    print()
    print("The data format should be:")
    print("  Vehicle_ID, Frame_ID, Total_Frames, Global_Time,")
    print("  Local_X, Local_Y, Global_X, Global_Y,")
    print("  v_length, v_Width, v_Class, v_Vel, v_Acc,")
    print("  Lane_ID, Preceding, Following, Space_Headway, Time_Headway")
    print("="*50)
    
    sample_data_path = os.path.join(args.output_dir, 'sample_trajectory.txt')
    print(f"\nCreating sample data file at: {sample_data_path}")
    
    sample_data = """1,1,100,1118847200.0,16.474,35.831,6042329.563,1873344.618,14.9,6.6,2,29.98,0.0,2,0,13,0.0,0.0
1,2,100,1118847200.1,16.534,36.532,6042329.949,1873345.209,14.9,6.6,2,30.11,1.3,2,0,13,0.0,0.0
1,3,100,1118847200.2,16.597,37.235,6042330.335,1873345.802,14.9,6.6,2,30.13,0.2,2,0,13,0.0,0.0
2,1,150,1118847200.0,24.474,42.831,6042337.563,1873350.618,15.2,6.8,2,28.45,0.0,3,0,14,0.0,0.0
2,2,150,1118847200.1,24.534,43.532,6042337.949,1873351.209,15.2,6.8,2,28.58,1.3,3,0,14,0.0,0.0"""
    
    with open(sample_data_path, 'w') as f:
        f.write(sample_data)
    
    print("Sample data file created successfully!")
    print("\nYou can test the model with this sample data:")
    print(f"  python train.py --data_path {sample_data_path} --epochs 1")


if __name__ == '__main__':
    main()