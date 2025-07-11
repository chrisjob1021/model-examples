from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import logging
import os

from datasets import DatasetDict, load_dataset, concatenate_datasets, load_from_disk

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DatasetProcessor:
    """
    A wrapper class to handle processing a specified dataset and saving to disk.
    
    This class provides a unified interface for dataset preprocessing, validation,
    and persistent storage with support for multiple output formats using HuggingFace datasets.
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        preprocess_fn: Callable,
        processor_name: Optional[str] = None,
        split_limits: Optional[Dict[str, int]] = None,
        num_threads: Optional[int] = None,
        start_index: Optional[int] = None,
        chunk_size: int = 50000,  # Process in chunks to prevent resource exhaustion
        batch_size: int = 500,    # Batch size for map operations (increased from 200)
        concatenate_only: bool = False,  # Only concatenate existing chunks, don't load original dataset
        features = None,  # Features schema for the output dataset
        **load_dataset_kwargs
    ):
        """
        Initializes the DatasetProcessor.

        Args:
            dataset_name: Name of the dataset to load using load_dataset
            output_dir: Directory where processed dataset will be saved
            processor_name: Name for this processor instance (used in file naming)
            preprocess_fn: Function to apply preprocessing to the dataset
            split_limits: Dictionary mapping split names to maximum number of features to process
            num_threads: Number of threads to use for processing (defaults to os.cpu_count())
            start_index: Index to start processing from (for debugging specific positions)
            chunk_size: Number of samples to process in each chunk
            batch_size: Batch size for map operations (increased from 200)
            concatenate_only: Only concatenate existing chunks, don't load original dataset
            features: Features schema for the output dataset (to preserve tensor format)
            **load_dataset_kwargs: Additional arguments to pass to load_dataset
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.processor_name = processor_name or "dataset_processor"
        self.preprocess_fn = preprocess_fn
        self.split_limits = split_limits or {}
        self.num_threads = num_threads or os.cpu_count()
        self.start_index = start_index
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.concatenate_only = concatenate_only
        self.features = features
        self.load_dataset_kwargs = load_dataset_kwargs
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        
        if self.concatenate_only:
            self.logger.info("Concatenate-only mode: skipping original dataset loading")
            self.dataset = None
        else:
            # Load the dataset using load_dataset
            self.logger.info(f"Loading dataset: {dataset_name}")
            self.logger.info(f"Using {self.num_threads} threads for processing")
            
            # Handle UTF-8 encoding errors that can occur with large datasets like ImageNet
            try:
                self.dataset = load_dataset(dataset_name, **load_dataset_kwargs)
            except UnicodeDecodeError as e:
                self.logger.warning(f"UTF-8 decode error encountered: {e}")
                self.logger.info("Retrying with ignore_verifications=True to handle encoding issues...")
                try:
                    # Retry with ignore_verifications to skip problematic files
                    self.dataset = load_dataset(dataset_name, ignore_verifications=True, **load_dataset_kwargs)
                    self.logger.info("Successfully loaded dataset with ignore_verifications=True")
                except Exception as e2:
                    self.logger.error(f"Failed to load dataset even with ignore_verifications: {e2}")
                    raise RuntimeError(f"Unable to load dataset {dataset_name} due to encoding issues. "
                                     f"Original error: {e}. Secondary error: {e2}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to prevent overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        
        # Initialize processed dataset
        self.processed_dataset = None
        
        # Check for existing progress to resume from
        self.resume_info = {}
        if self.chunk_size > 0:
            self._check_resume_progress()

    def _check_resume_progress(self):
        """Check for existing chunk progress and determine where to resume from."""
        progress_dir = self.output_dir / f"{self.processor_name}_progress"
        
        if not progress_dir.exists():
            self.logger.info("No previous progress found - starting fresh")
            return
        
        self.logger.info(f"Checking for previous progress in {progress_dir}")
        
        # Find all chunk files for each split
        for split_name in ["train", "validation", "test"]:
            chunk_files = list(progress_dir.glob(f"{split_name}_chunk_*"))
            
            if chunk_files:
                # Extract chunk numbers and sort them
                chunk_info = []
                for chunk_file in chunk_files:
                    try:
                        chunk_num = int(chunk_file.name.split('_chunk_')[1])
                        chunk_info.append((chunk_num, chunk_file))
                    except (IndexError, ValueError):
                        continue
                
                if chunk_info:
                    # Sort by chunk number and find the maximum
                    chunk_info.sort(key=lambda x: x[0])
                    max_chunk_num = max(chunk_num for chunk_num, _ in chunk_info)
                    max_chunk_file = next(chunk_file for chunk_num, chunk_file in chunk_info if chunk_num == max_chunk_num)
                    
                    # Try to load the maximum chunk first
                    try:
                        from datasets import load_from_disk
                        test_dataset = load_from_disk(str(max_chunk_file))
                        expected_samples = self.chunk_size
                        actual_samples = len(test_dataset)
                        
                        self.logger.info(f"Loaded max chunk {max_chunk_num}: {actual_samples} samples")
                        
                        # Check if chunk appears corrupted (legitimately small final chunks are normal)
                        # We don't know total dataset size here, so we'll be very lenient
                        # Only flag chunks that are suspiciously small (likely corrupted)
                        if actual_samples < expected_samples * 0.1:  # Only flag if less than 10% of expected
                            self.logger.warning(f"Chunk {max_chunk_num} appears corrupted: "
                                              f"{actual_samples} samples (expected ~{expected_samples})")
                            raise ValueError(f"Corrupted chunk detected - too few samples")
                        elif actual_samples < expected_samples * 0.8:
                            # Log info about smaller chunks, but don't treat as error
                            self.logger.info(f"Chunk {max_chunk_num} is smaller than expected: "
                                           f"{actual_samples} samples (expected ~{expected_samples}) - "
                                           f"likely final chunk or processing variation")
                        
                        # If max chunk loads successfully and appears complete, use it for resume
                        self.resume_info[split_name] = {
                            'latest_chunk': max_chunk_num,
                            'chunk_files': [max_chunk_file]
                        }
                        self.logger.info(f"Will resume from chunk {max_chunk_num + 1}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load or validate max chunk {max_chunk_num}: {e}")
                        
                        # Delete the corrupted/incomplete max chunk
                        try:
                            import shutil
                            shutil.rmtree(max_chunk_file)
                            self.logger.info(f"Deleted corrupted/incomplete chunk: {max_chunk_file}")
                        except Exception as delete_error:
                            self.logger.warning(f"Could not delete corrupted chunk {max_chunk_file}: {delete_error}")
                        
                        # Try to load the previous chunk
                        if len(chunk_info) > 1:
                            prev_chunk_num = max_chunk_num - 1
                            prev_chunk_file = next((chunk_file for chunk_num, chunk_file in chunk_info if chunk_num == prev_chunk_num), None)
                            
                            if prev_chunk_file:
                                try:
                                    test_dataset = load_from_disk(str(prev_chunk_file))
                                    self.logger.info(f"Successfully loaded previous chunk {prev_chunk_num}: {len(test_dataset)} samples")
                                    
                                    # Use the previous chunk for resume (will reprocess the corrupted chunk)
                                    self.resume_info[split_name] = {
                                        'latest_chunk': prev_chunk_num,
                                        'chunk_files': [prev_chunk_file]
                                    }
                                    self.logger.info(f"Will resume from chunk {prev_chunk_num + 1} (reprocessing corrupted chunk)")
                                    
                                except Exception as e2:
                                    self.logger.warning(f"Failed to load previous chunk {prev_chunk_num}: {e2}")
                                    # Don't try any other chunks - just skip this split
        
        if self.resume_info:
            self.logger.info(f"Resume info: {list(self.resume_info.keys())}")
        else:
            self.logger.info("No valid progress chunks found")

    def _apply_preprocessing(self) -> None:
        """Apply preprocessing function to the dataset using HuggingFace datasets."""
        if self.concatenate_only:
            self.logger.info("Concatenate-only mode: loading and combining existing chunks")
            
            # Process each split using the single concatenation method
            progress_dir = self.output_dir / f"{self.processor_name}_progress"
            
            if not progress_dir.exists():
                self.logger.error(f"Progress directory not found: {progress_dir}")
                self.logger.error("Cannot concatenate chunks - no progress files exist")
                return
            
            self.logger.info(f"Loading existing chunks from: {progress_dir}")
            self.processed_dataset = DatasetDict()
            
            for split_name in ["train", "validation", "test"]:
                self.logger.info(f"Processing {split_name} split...")
                
                split_dataset = self._concatenate_all_chunks_for_split(split_name)
                
                if split_dataset is not None:
                    self.processed_dataset[split_name] = split_dataset
                    
                    # Show sample structure
                    if len(split_dataset) > 0:
                        sample = split_dataset[0]
                        self.logger.info(f"Sample structure: {list(sample.keys())}")
            
            if not self.processed_dataset:
                self.logger.error("No datasets were successfully concatenated")
            else:
                self.logger.info(f"Concatenation completed for {len(self.processed_dataset)} splits")
            return
            
        self.logger.info("Applying preprocessing function...")
        
        # Centralize map arguments
        map_kwargs = {
            'function': self.preprocess_fn,
            'batched': True,
            'batch_size': self.batch_size,
            'num_proc': self.num_threads,
            'load_from_cache_file': False,  # Disable caching to prevent huge cache files
            'cache_file_name': None,        # Don't create cache files
        }
        
        # Add features schema if provided (to preserve tensor format)
        if self.features is not None:
            map_kwargs['features'] = self.features

        if isinstance(self.dataset, DatasetDict):
            # Process each split separately
            self.processed_dataset = DatasetDict()
            for split_name, split_dataset in self.dataset.items():
                self.logger.info(f"Processing split: {split_name}")
                
                # Apply split limit if specified
                if split_name in self.split_limits and self.split_limits[split_name] is not None:
                    limit = self.split_limits[split_name]
                    
                    # Skip split if limit is 0
                    if limit == 0:
                        self.logger.info(f"Skipping {split_name} (limit set to 0)")
                        continue
                        
                    self.logger.info(f"Limiting {split_name} to {limit} features")
                    split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))
                
                # Apply start_index if specified (for debugging specific positions)
                if self.start_index is not None:
                    self.logger.info(f"Starting {split_name} from index {self.start_index}")
                    split_dataset = split_dataset.select(range(self.start_index, len(split_dataset)))
                
                # Process in chunks to prevent resource exhaustion
                processed_split = self._process_split_in_chunks(
                    split_dataset, split_name, map_kwargs
                )
                
                # Only add to processed_dataset if processing was successful
                if processed_split is not None and len(processed_split) > 0:
                    self.processed_dataset[split_name] = processed_split
                    self.logger.info(f"Successfully processed {split_name}: {len(processed_split)} samples")
                else:
                    self.logger.warning(f"Skipping {split_name} - no valid samples processed")
        else:
            # Process single dataset in chunks
            self.processed_dataset = self._process_split_in_chunks(
                self.dataset, "dataset", map_kwargs
            )
        
        self.logger.info("Preprocessing completed successfully")

    def _process_split_in_chunks(self, dataset, split_name, map_kwargs):
        """Process a dataset split in chunks to prevent resource exhaustion."""
        import gc
        import psutil
        from datasets import concatenate_datasets, load_from_disk
        
        total_samples = len(dataset)
        processed_chunks = []
        
        # Check if we can resume from previous progress
        start_chunk = 1
        if split_name in self.resume_info:
            latest_chunk = self.resume_info[split_name]['latest_chunk']
            chunk_files = self.resume_info[split_name]['chunk_files']
            
            self.logger.info(f"Resuming {split_name} from chunk {latest_chunk}")
            
            # Load only the latest chunk
            if len(chunk_files) == 1:
                chunk_file = chunk_files[0]
                try:
                    chunk_dataset = load_from_disk(str(chunk_file))
                    if len(chunk_dataset) > 0:
                        processed_chunks.append(chunk_dataset)
                        self.logger.info(f"Loaded latest chunk {latest_chunk}: {len(chunk_dataset)} samples")
                        
                        # Calculate where to start processing (continue from after the loaded chunk)
                        start_chunk = latest_chunk + 1
                        start_sample = latest_chunk * self.chunk_size
                        
                        # Check if we still have more samples to process
                        if start_sample >= total_samples:
                            self.logger.info(f"{split_name} already fully processed - loading all chunks for combination")
                            
                            # Use consolidated concatenation method
                            final_dataset = self._concatenate_all_chunks_for_split(split_name)
                            return final_dataset if final_dataset is not None else chunk_dataset
                        
                        self.logger.info(f"Will continue processing from chunk {start_chunk} (sample {start_sample})")
                    else:
                        self.logger.warning(f"Latest chunk is empty")
                        start_chunk = latest_chunk
                except Exception as e:
                    self.logger.error(f"Failed to load latest chunk: {e}")
                    start_chunk = latest_chunk
            else:
                self.logger.warning("No chunk files found in resume info")
                start_chunk = 1
        else:
            self.logger.info(f"Starting {split_name} from beginning")
        
        self.logger.info(f"Processing {split_name} in chunks of {self.chunk_size} samples")
        self.logger.info(f"Total samples to process: {total_samples}")
        
        # Calculate the starting point for processing
        start_sample = (start_chunk - 1) * self.chunk_size
        
        for chunk_start in range(start_sample, total_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_samples)
            chunk_num = chunk_start // self.chunk_size + 1
            total_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
            
            self.logger.info(f"Processing {split_name} chunk {chunk_num}/{total_chunks} "
                           f"(samples {chunk_start}-{chunk_end-1})")
            
            # Monitor memory usage
            memory_percent = psutil.virtual_memory().percent
            self.logger.info(f"Memory usage before chunk: {memory_percent:.1f}%")
            
            try:
                # Select chunk with debugging
                self.logger.info(f"Selecting chunk data: indices {chunk_start} to {chunk_end-1}")
                try:
                    chunk_dataset = dataset.select(range(chunk_start, chunk_end))
                    self.logger.info(f"Successfully selected chunk dataset: {len(chunk_dataset)} samples")
                except Exception as select_error:
                    self.logger.error(f"Error during dataset selection: {select_error}")
                    if "utf-8" in str(select_error).lower():
                        self.logger.error(f"UTF-8 error during dataset selection - problematic indices: {chunk_start}-{chunk_end-1}")
                    raise select_error
                
                # Process chunk with debugging
                self.logger.info(f"Starting map operation for chunk {chunk_num}")
                try:
                    processed_chunk = chunk_dataset.map(
                        desc=f"Preprocessing {split_name} chunk {chunk_num}",
                        remove_columns=chunk_dataset.column_names,
                        **map_kwargs
                    )
                    self.logger.info(f"Successfully completed map operation for chunk {chunk_num}")
                except Exception as map_error:
                    self.logger.error(f"Error during map operation: {map_error}")
                    if "utf-8" in str(map_error).lower():
                        self.logger.error(f"UTF-8 error during map operation - chunk indices: {chunk_start}-{chunk_end-1}")
                    raise map_error
                
                # print(type(processed_chunk))
                print(type(processed_chunk['labels']))
                print(type(processed_chunk['pixel_values']))

                # No need to filter - preprocessing only returns successful images
                processed_chunks.append(processed_chunk)
                
                # Save progress after each chunk - save only the new chunk (not cumulative)
                if processed_chunk:
                    self._save_chunk_progress(processed_chunk, split_name, chunk_num)
                    
                    self.logger.info(f"Chunk {chunk_num} completed: {len(processed_chunk)} new samples processed")
                    
                    # Calculate total processed including resumed chunks
                    total_processed = sum(len(chunk) for chunk in processed_chunks)
                    self.logger.info(f"Total processed so far: {total_processed} samples")
                
                # Force garbage collection after each chunk
                gc.collect()
                
                # Monitor memory after processing
                memory_percent = psutil.virtual_memory().percent
                self.logger.info(f"Memory usage after chunk: {memory_percent:.1f}%")
                
            except Exception as e:
                import traceback
                
                self.logger.error(f"Error processing chunk {chunk_num}: {e}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Full traceback:")
                
                # Log the full traceback to see exactly where the error occurs
                tb_lines = traceback.format_exc().split('\n')
                for line in tb_lines:
                    if line.strip():
                        self.logger.error(f"  {line}")
                
                # Additional debugging for UTF-8 errors
                if "utf-8" in str(e).lower() or "codec" in str(e).lower():
                    self.logger.error(f"UTF-8 decoding error detected!")
                    self.logger.error(f"Chunk range: samples {chunk_start} to {chunk_end-1}")
                    self.logger.error(f"This suggests a problematic file in the dataset at these indices")
                
                self.logger.info("Error occurred - progress has already been saved for successful chunks")
                
                # Don't try to save anything here - successful chunks were already saved
                # The failed chunk should not be saved, and previous chunks are already on disk
                self.logger.info(f"Chunks 1-{chunk_num-1} were already saved successfully")
                self.logger.info(f"Chunk {chunk_num} failed and will be skipped")
                
                # Re-raise the error
                raise e
        
        # Combine all chunks using consolidated method
        if processed_chunks:
            self.logger.info(f"Processing completed - combining all chunks for {split_name}")
            final_dataset = self._concatenate_all_chunks_for_split(split_name)
            
            if final_dataset is not None:
                self.logger.info(f"Completed processing {split_name}: {len(final_dataset)} total samples")
                return final_dataset
            else:
                self.logger.warning(f"No valid chunks found for {split_name}")
                return None
        else:
            self.logger.warning(f"No samples were successfully processed for {split_name}")
            return None

    def _save_chunk_progress(self, dataset, split_name, chunk_num):
        """Save progress after processing each chunk."""
        try:
            progress_dir = self.output_dir / f"{self.processor_name}_progress"
            progress_dir.mkdir(exist_ok=True)
            
            progress_file = progress_dir / f"{split_name}_chunk_{chunk_num}"
            
            # Check if chunk already exists (shouldn't happen in normal operation)
            if progress_file.exists():
                self.logger.warning(f"Chunk {chunk_num} already exists at {progress_file}, skipping save")
                return
            
            dataset.save_to_disk(str(progress_file))
            
            self.logger.info(f"Saved progress for {split_name} chunk {chunk_num} to {progress_file}")
        except Exception as e:
            self.logger.warning(f"Could not save chunk progress: {e}")

    def _cleanup_progress_files(self):
        """Clean up progress files after successful completion."""
        progress_dir = self.output_dir / f"{self.processor_name}_progress"
        
        if progress_dir.exists():
            try:
                import shutil
                shutil.rmtree(progress_dir)
                self.logger.info(f"Cleaned up progress files from {progress_dir}")
            except Exception as e:
                self.logger.warning(f"Could not clean up progress files: {e}")

    def cleanup_all_progress(self):
        """Manually clean up all progress files (useful for corrupted chunks)."""
        progress_dir = self.output_dir / f"{self.processor_name}_progress"
        
        if progress_dir.exists():
            try:
                import shutil
                shutil.rmtree(progress_dir)
                self.logger.info(f"Manually cleaned up all progress files from {progress_dir}")
                # Reset resume info
                self.resume_info = {}
                return True
            except Exception as e:
                self.logger.error(f"Could not clean up progress files: {e}")
                return False
        else:
            self.logger.info("No progress files to clean up")
            return True

    def _save_dataset(self) -> Dict[str, str]:
        """Save the processed dataset to disk using HuggingFace datasets."""
        saved_files = {}
        
        # Check if processed dataset exists and is not None
        if self.processed_dataset is None:
            self.logger.warning("No processed dataset to save - processing may have failed")
            return saved_files
        
        self.logger.info("Saving dataset using HuggingFace save_to_disk...")
        
        if isinstance(self.processed_dataset, DatasetDict):
            # Check if DatasetDict has any valid splits
            valid_splits = {k: v for k, v in self.processed_dataset.items() if v is not None and len(v) > 0}
            
            if not valid_splits:
                self.logger.warning("No valid splits found in DatasetDict - nothing to save")
                return saved_files
            
            # Save entire DatasetDict to disk
            filename = f"{self.processor_name}"
            filepath = self.output_dir / filename
            
            # Create a new DatasetDict with only valid splits
            if len(valid_splits) < len(self.processed_dataset):
                filtered_dataset = DatasetDict(valid_splits)
                filtered_dataset.save_to_disk(str(filepath))
                self.logger.info(f"Saved {len(valid_splits)} valid splits (filtered from {len(self.processed_dataset)} total)")
            else:
                self.processed_dataset.save_to_disk(str(filepath))
                
            saved_files["dataset_dict"] = str(filepath)
        else:
            # Single dataset - check if it's valid
            if len(self.processed_dataset) == 0:
                self.logger.warning("Processed dataset is empty - nothing to save")
                return saved_files
                
            # Save single dataset to disk
            filename = f"{self.processor_name}"
            filepath = self.output_dir / filename
            self.processed_dataset.save_to_disk(str(filepath))
            saved_files["dataset"] = str(filepath)
        
        self.logger.info(f"Saved dataset to: {filepath}")
        return saved_files

    def _concatenate_all_chunks_for_split(self, split_name):
        """Concatenate all existing chunks for a specific split."""
        progress_dir = self.output_dir / f"{self.processor_name}_progress"
        
        if not progress_dir.exists():
            self.logger.warning(f"Progress directory not found: {progress_dir}")
            return None
        
        # Find chunk files for this split
        chunk_files = list(progress_dir.glob(f"{split_name}_chunk_*"))
        
        if not chunk_files:
            self.logger.info(f"No chunks found for {split_name}")
            return None
        
        # Sort chunks by number
        chunk_info = []
        for chunk_file in chunk_files:
            try:
                chunk_num = int(chunk_file.name.split('_chunk_')[1])
                chunk_info.append((chunk_num, chunk_file))
            except (IndexError, ValueError):
                self.logger.warning(f"Invalid chunk name: {chunk_file.name}")
                continue
        
        if not chunk_info:
            self.logger.warning(f"No valid chunks found for {split_name}")
            return None
        
        # Sort by chunk number
        chunk_info.sort(key=lambda x: x[0])
        chunk_numbers = [num for num, _ in chunk_info]
        
        self.logger.info(f"Found {split_name} chunks: {chunk_numbers}")
        
        # Load and concatenate chunks
        chunks = []
        total_samples = 0
        
        for chunk_num, chunk_file in chunk_info:
            try:
                self.logger.info(f"Loading {split_name} chunk {chunk_num}...")
                chunk_dataset = load_from_disk(str(chunk_file))
                chunk_samples = len(chunk_dataset)
                
                if chunk_samples > 0:
                    chunks.append(chunk_dataset)
                    total_samples += chunk_samples
                    self.logger.info(f"Loaded {chunk_samples:,} samples from chunk {chunk_num}")
                else:
                    self.logger.warning(f"Chunk {chunk_num} is empty")
                    
            except Exception as e:
                self.logger.error(f"Failed to load chunk {chunk_num}: {e}")
                continue
        
        if chunks:
            self.logger.info(f"Concatenating {len(chunks)} chunks for {split_name}...")
            final_dataset = concatenate_datasets(chunks)
            self.logger.info(f"Successfully concatenated {split_name}: {len(final_dataset):,} total samples")
            return final_dataset
        else:
            self.logger.warning(f"No valid chunks loaded for {split_name}")
            return None

    def process(self) -> Dict[str, str]:
        """
        Process the dataset with the given preprocessing function.
            
        Returns:
            Dictionary with paths to saved dataset files
        """
        self.logger.info(f"Starting dataset processing with {self.processor_name}")
        
        # Check if we have resume information
        if self.resume_info:
            self.logger.info("Resume mode detected - will continue from latest saved chunks")
            for split_name, info in self.resume_info.items():
                self.logger.info(f"  {split_name}: resuming from chunk {info['latest_chunk']}")
        else:
            self.logger.info("Starting fresh processing")
            
        if self.start_index is not None and self.start_index > 0:
            self.logger.info(f"Debug mode: starting from index {self.start_index}")
        
        # Apply preprocessing
        self._apply_preprocessing()
        
        # Save the processed dataset
        results = self._save_dataset()
        
        # Clean up progress files only if explicitly requested
        # self._cleanup_progress_files()  # Commented out to preserve chunks
        
        return results