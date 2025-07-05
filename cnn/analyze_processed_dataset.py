#!/usr/bin/env python3
"""Analyze the processed dataset and verify it against original chunks"""

from pathlib import Path
from datasets import load_from_disk, load_dataset

def analyze_processed_dataset():
    """Analyze the final processed dataset and verify against chunks."""
    
    # We'll determine the actual chunk size from the first chunk we find
    chunk_size = None
    total_chunk_samples = 0

    # Check for the final processed dataset
    processed_dataset_path = Path("./processed_datasets/imagenet_processor")
    progress_dir = Path("./processed_datasets/imagenet_processor_progress")
    
    print("🔍 Analyzing processed dataset and chunks")
    print("=" * 60)
    
    # Load original dataset for comparison
    print("📥 Loading original dataset for alignment check...")
    try:
        original_dataset = load_dataset("imagenet-1k", trust_remote_code=True)
        print("✅ Original dataset loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load original dataset: {e}")
        original_dataset = None
    
    # Check if final processed dataset exists
    if processed_dataset_path.exists():
        print(f"\n📊 FINAL PROCESSED DATASET:")
        print(f"  Path: {processed_dataset_path}")
        
        try:
            # Load the final processed dataset
            final_dataset = load_from_disk(str(processed_dataset_path))
            
            if hasattr(final_dataset, 'items'):
                # DatasetDict
                print(f"  🔄 DatasetDict with splits:")
                for split_name, split_dataset in final_dataset.items():
                    print(f"    📂 {split_name}: {len(split_dataset):,} samples")
                    
                    # Verify against original dataset if available
                    if original_dataset and split_name in original_dataset:
                        original_split = original_dataset[split_name]
                        
                        # Check alignment at key positions
                        test_indices = [0, min(1000, len(split_dataset)-1), len(split_dataset)-1] if len(split_dataset) > 0 else []
                        
                        alignment_ok = True
                        for test_idx in test_indices:
                            if test_idx < len(split_dataset) and test_idx < len(original_split):
                                try:
                                    final_label = split_dataset[test_idx]['labels']
                                    original_label = original_split[test_idx]['label']
                                    
                                    if final_label != original_label:
                                        print(f"      ❌ Label mismatch at idx {test_idx}: final={final_label}, original={original_label}")
                                        alignment_ok = False
                                        break
                                except Exception as e:
                                    print(f"      ⚠️  Could not verify alignment at idx {test_idx}: {e}")
                                    break
                        
                        if alignment_ok and test_indices:
                            print(f"      ✅ Final dataset alignment verified")
                        elif not test_indices:
                            print(f"      ⚠️  Empty dataset - no alignment to verify")
                        
                        # Show processing progress
                        original_samples = len(original_split)
                        processed_samples = len(split_dataset)
                        progress_pct = (processed_samples / original_samples) * 100
                        print(f"      📈 Processing progress: {progress_pct:.1f}% ({processed_samples:,}/{original_samples:,})")
            else:
                # Single dataset
                print(f"  📂 Single dataset: {len(final_dataset):,} samples")
                
                # Show features
                if hasattr(final_dataset, 'features'):
                    print(f"  🔧 Features: {list(final_dataset.features.keys())}")
            
            print(f"  ✅ Successfully loaded final processed dataset")
            
        except Exception as e:
            print(f"  ❌ Failed to load final processed dataset: {e}")
    else:
        print(f"\n❌ No final processed dataset found at {processed_dataset_path}")
    
    # Analyze individual chunks if they exist
    if progress_dir.exists():
        print(f"\n📊 INDIVIDUAL CHUNKS ANALYSIS:")
        print(f"  Progress directory: {progress_dir}")
        
        for split_name in ["train", "validation", "test"]:
            chunk_files = list(progress_dir.glob(f"{split_name}_chunk_*"))
            
            if not chunk_files:
                print(f"\n  {split_name.upper()}: No chunks found")
                continue
            
            print(f"\n  {split_name.upper()} CHUNKS:")
            
            # Find all chunks and sort them
            chunk_info = []
            for chunk_file in chunk_files:
                try:
                    chunk_num = int(chunk_file.name.split('_chunk_')[1])
                    chunk_info.append((chunk_num, chunk_file))
                except (IndexError, ValueError):
                    print(f"    ⚠️  Invalid chunk name: {chunk_file.name}")
                    continue
            
            if not chunk_info:
                print(f"    No valid chunks found")
                continue
            
            # Sort chunks by number
            chunk_info.sort(key=lambda x: x[0])
            
            print(f"    Found chunks: {[num for num, _ in chunk_info]}")
            
            # Analyze each individual chunk
            running_total = 0  # Track cumulative samples processed
            
            for chunk_num, chunk_file in chunk_info:
                try:
                    print(f"      📦 Loading chunk {chunk_num}...")
                    chunk_dataset = load_from_disk(str(chunk_file))
                    chunk_samples = len(chunk_dataset)
                    total_chunk_samples += chunk_samples
                    
                    print(f"         💾 Loaded: {chunk_samples:,} samples")
                    print(f"         📁 File: {chunk_file}")
                    
                    # Determine chunk size from first chunk if not set
                    if chunk_size is None:
                        chunk_size = chunk_samples
                        print(f"         📏 Detected chunk size: {chunk_size:,}")
                    
                    # Show features
                    if hasattr(chunk_dataset, 'features'):
                        print(f"         🔧 Features: {list(chunk_dataset.features.keys())}")
                    
                    # Check if chunk size matches expectations
                    if chunk_samples == chunk_size:
                        print(f"         ✅ Expected size ({chunk_size:,})")
                    else:
                        print(f"         ⚠️  Unexpected size (expected {chunk_size:,}, got {chunk_samples:,})")
                    
                    # Verify data structure
                    if chunk_samples > 0:
                        sample = chunk_dataset[0]
                        print(f"         📊 Sample structure: {list(sample.keys())}")
                        
                        # # Check data types
                        # if 'pixel_values' in sample:
                        #     pixel_shape = sample['pixel_values'].shape if hasattr(sample['pixel_values'], 'shape') else "Unknown"
                        #     print(f"         🖼️  Image shape: {pixel_shape}")
                        
                        if 'labels' in sample:
                            label_value = sample['labels']
                            print(f"         🏷️  Label type: {type(label_value)}, value: {label_value}")
                    
                    # Verify alignment with original dataset if available
                    if original_dataset and split_name in original_dataset:
                        original_split = original_dataset[split_name]
                        
                        # Calculate expected start position for this chunk
                        # Use running total instead of chunk_num * chunk_size since chunks might be variable size
                        expected_start = running_total
                        print(f"         🎯 Expected position: samples {expected_start:,} - {expected_start + chunk_samples - 1:,}")
                        
                        # Check multiple samples for alignment
                        sample_indices = []
                        if chunk_samples > 0:
                            sample_indices.append(0)  # First sample
                            if chunk_samples > 1:
                                sample_indices.append(chunk_samples - 1)  # Last sample
                            if chunk_samples > 10:
                                sample_indices.append(chunk_samples // 2)  # Middle sample
                        
                        alignment_ok = True
                        alignment_details = []
                        
                        for local_idx in sample_indices:
                            global_idx = expected_start + local_idx
                            
                            if global_idx < len(original_split) and local_idx < chunk_samples:
                                try:
                                    chunk_label = chunk_dataset[local_idx]['labels']
                                    original_label = original_split[global_idx]['label']
                                    
                                    if chunk_label == original_label:
                                        alignment_details.append(f"idx {local_idx}→{global_idx}: ✅ {chunk_label}")
                                    else:
                                        alignment_details.append(f"idx {local_idx}→{global_idx}: ❌ chunk={chunk_label}, orig={original_label}")
                                        alignment_ok = False
                                except Exception as e:
                                    alignment_details.append(f"idx {local_idx}→{global_idx}: ⚠️  Error: {e}")
                                    alignment_ok = False
                        
                        # Show alignment results
                        print(f"         🔍 Alignment check:")
                        for detail in alignment_details:
                            print(f"           {detail}")
                        
                        if alignment_ok and sample_indices:
                            print(f"         ✅ Chunk alignment verified ({len(sample_indices)} samples checked)")
                        elif not sample_indices:
                            print(f"         ⚠️  Empty chunk - no alignment to verify")
                        else:
                            print(f"         ❌ Chunk alignment failed")
                    
                    # Update running total for next chunk
                    running_total += chunk_samples
                    
                    print(f"         ✅ Chunk {chunk_num} analysis complete")
                    print()  # Add spacing between chunks
               
                except Exception as e:
                    print(f"      ❌ Failed to load chunk {chunk_num}: {e}")
                    print(f"         📁 File: {chunk_file}")
                    print(f"         🔍 Error type: {type(e).__name__}")
                    print()
            
            print(f"    📊 Summary:")
            print(f"      Total chunks: {len(chunk_info)}")
            print(f"      Total samples in chunks: {total_chunk_samples:,}")
            if chunk_size is not None:
                print(f"      Expected total (if uniform): {len(chunk_info) * chunk_size:,}")
            else:
                print(f"      Expected total: Unknown (no chunks loaded)")
            
            # Compare chunk total with final dataset
            if processed_dataset_path.exists():
                try:
                    final_dataset = load_from_disk(str(processed_dataset_path))
                    if hasattr(final_dataset, 'items') and split_name in final_dataset:
                        final_samples = len(final_dataset[split_name])
                        
                        if final_samples == total_chunk_samples:
                            print(f"      ✅ Final dataset matches chunk total ({final_samples:,})")
                        else:
                            print(f"      ❌ Final dataset mismatch: final={final_samples:,}, chunks={total_chunk_samples:,}")
                except Exception as e:
                    print(f"      ⚠️  Could not compare with final dataset: {e}")
    else:
        print(f"\n📊 No progress directory found - no individual chunks to analyze")
    
    print("\n" + "=" * 60)
    print("💡 Analysis complete!")

if __name__ == "__main__":
    analyze_processed_dataset() 