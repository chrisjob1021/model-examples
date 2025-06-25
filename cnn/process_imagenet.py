#!/usr/bin/env python3
"""Process ImageNet-1k dataset using DatasetProcessor"""

import torch
from prelu_cnn import preprocess_images

# Import from shared_utils package
from shared_utils import DatasetProcessor

def main():
    """Process ImageNet-1k dataset using DatasetProcessor."""
    
    print("🚀 Processing ImageNet-1k dataset using DatasetProcessor")
    print("=" * 60)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Define preprocessing function
    print("📝 Setting up preprocessing function...")
    
    # Create DatasetProcessor for ImageNet-1k
    print("🔄 Creating DatasetProcessor for ImageNet-1k...")
    
    processor = DatasetProcessor(
        dataset_name="imagenet-1k",
        output_dir="./processed_datasets",
        preprocess_fn=preprocess_images,
        processor_name="imagenet_processor",
        split_limits={
            "train": None,
            "validation": None,
            "test": None
        },
        num_threads=2,
        trust_remote_code=True,       # Trust remote code for dataset loading
    )
    
    print(f"✅ DatasetProcessor created successfully")
    print(f"📁 Output directory: {processor.output_dir}")
    print(f"📊 Split limits: {processor.split_limits}")
    
    # Process the dataset
    print(f"\n🔄 Processing ImageNet-1k dataset...")
    print("This may take a while for the first run as the dataset needs to be downloaded...")
    
    try:
        results = processor.process()
        
        print(f"\n✅ Dataset processing completed successfully!")
        print(f"📊 Processing Results:")
        print(f"  Processor name: {results['processor_name']}")
        print(f"  Dataset name: {results['dataset_name']}")
        print(f"  Timestamp: {results['timestamp']}")
        print(f"  Output directory: {results['output_dir']}")
        print(f"  Saved files: {results['saved_files']}")
        print(f"  Split limits: {results['split_limits']}")
        print(f"  Success: {results['success']}")
        
        # Display dataset information
        print(f"\n📈 Dataset Information:")
        if hasattr(processor, 'processed_dataset'):
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
        
        print(f"\n💾 Processed dataset saved to: {results['output_dir']}")
        print(f"📁 You can now use this processed dataset for training!")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        print(f"💡 Make sure you have access to the ImageNet-1k dataset")
        print(f"💡 You may need to download it manually or have proper credentials")
        return
    
    print(f"\n✅ ImageNet-1k processing completed successfully!")
    print(f"📁 Check {results['output_dir']} for processed dataset files")

if __name__ == "__main__":
    main() 