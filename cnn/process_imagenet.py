#!/usr/bin/env python3
"""Process ImageNet-1k dataset using DatasetProcessor"""

import torch
import logging
import argparse
from prelu_cnn import preprocess_images

# Import from shared_utils package
from shared_utils import DatasetProcessor

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(description="Process ImageNet-1k dataset")
    parser.add_argument(
        "--concatenate-only", 
        action="store_true",
        help="Only concatenate existing progress chunks, don't process new data"
    )
    
    args = parser.parse_args()
    
    if args.concatenate_only:
        print("🔄 Concatenating existing progress chunks using DatasetProcessor...")
        print("=" * 60)
    else:
        print("🚀 Processing ImageNet-1k dataset using DatasetProcessor")
        print("=" * 60)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create DatasetProcessor for ImageNet-1k
    print("🔄 Creating DatasetProcessor for ImageNet-1k...")
    
    processor = DatasetProcessor(
        dataset_name="imagenet-1k",
        output_dir="./processed_datasets",
        preprocess_fn=preprocess_images,
        processor_name="imagenet_processor",
        split_limits={
            "train": 100000,
            "validation": 0,
            "test": 0
        },
        # PERFORMANCE OPTIMIZATIONS:
        num_threads=2,
        chunk_size=250000,
        batch_size=200,
        trust_remote_code=True,
        cache_dir=None,  # Don't use cache
        concatenate_only=args.concatenate_only,  # Skip dataset loading if concatenate-only
        # download_mode="force_redownload",  # Force redownload of the dataset
    )
    
    print(f"✅ DatasetProcessor created successfully")
    print(f"📁 Output directory: {processor.output_dir}")
    print(f"📊 Split limits: {processor.split_limits}")
    
    if args.concatenate_only:
        print(f"\n🔗 Concatenate-only mode: will skip dataset loading and only combine existing chunks")
        print(f"💡 No processing of original ImageNet data will be performed")
    else:
        print(f"\n🔄 Processing ImageNet-1k dataset...")
        print(f"📥 This will load the original dataset and apply preprocessing")
    
    try:
        results = processor.process()
        
        if args.concatenate_only:
            print(f"\n✅ Chunk concatenation completed successfully!")
        else:
            print(f"\n✅ Dataset processing completed successfully!")
            
        print(f"📊 Processing Results:")
        print(f"  Saved files: {results}")
        
        # Display dataset information
        print(f"\n📈 Dataset Information:")
        if hasattr(processor, 'processed_dataset') and processor.processed_dataset:
            if hasattr(processor.processed_dataset, 'items'):
                # DatasetDict
                for split_name, split_dataset in processor.processed_dataset.items():
                    print(f"  {split_name}: {len(split_dataset)} samples")
                    if len(split_dataset) > 0:
                        print(f"    Features: {list(split_dataset.features.keys())}")
                        print(f"    Sample keys: {list(split_dataset[0].keys())}")
            else:
                # Single Dataset
                print(f"  Total samples: {len(processor.processed_dataset)}")
                print(f"  Features: {list(processor.processed_dataset.features.keys())}")
                if len(processor.processed_dataset) > 0:
                    print(f"  Sample keys: {list(processor.processed_dataset[0].keys())}")
        
        print(f"\n💾 Processed dataset saved to: {processor.output_dir}")
        print(f"📁 You can now use this processed dataset for training!")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return
    
    if args.concatenate_only:
        print(f"\n✅ ImageNet-1k chunk concatenation completed successfully!")
    else:
        print(f"\n✅ ImageNet-1k processing completed successfully!")
    print(f"📁 Check {processor.output_dir} for processed dataset files")

if __name__ == "__main__":
    main() 