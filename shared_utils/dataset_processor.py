from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import logging
import os

from datasets import DatasetDict, load_dataset, concatenate_datasets

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
        self.load_dataset_kwargs = load_dataset_kwargs
        
        # Load the dataset using load_dataset
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
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

    def _apply_preprocessing(self) -> None:
        """Apply preprocessing function to the dataset using HuggingFace datasets."""
        self.logger.info("Applying preprocessing function...")
        
        # Centralize map arguments
        map_kwargs = {
            'function': self.preprocess_fn,
            'batched': True,
            'batch_size': 200,
            'num_proc': self.num_threads,
            'load_from_cache_file': False,
        }

        if isinstance(self.dataset, DatasetDict):
            # Process each split separately
            self.processed_dataset = DatasetDict()
            for split_name, split_dataset in self.dataset.items():
                self.logger.info(f"Processing split: {split_name}")
                
                # Apply split limit if specified
                if split_name in self.split_limits and self.split_limits[split_name] is not None:
                    limit = self.split_limits[split_name]
                    self.logger.info(f"Limiting {split_name} to {limit} features")
                    split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))
                
                # Apply start_index if specified (for debugging specific positions)
                if self.start_index is not None:
                    self.logger.info(f"Starting {split_name} from index {self.start_index}")
                    split_dataset = split_dataset.select(range(self.start_index, len(split_dataset)))
                
                # Process in chunks to prevent resource exhaustion
                self.processed_dataset[split_name] = self._process_split_in_chunks(
                    split_dataset, split_name, map_kwargs
                )
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
        from datasets import concatenate_datasets
        
        total_samples = len(dataset)
        processed_chunks = []
        
        self.logger.info(f"Processing {split_name} in chunks of {self.chunk_size} samples")
        self.logger.info(f"Total samples to process: {total_samples}")
        
        for chunk_start in range(0, total_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_samples)
            chunk_num = chunk_start // self.chunk_size + 1
            total_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
            
            self.logger.info(f"Processing {split_name} chunk {chunk_num}/{total_chunks} "
                           f"(samples {chunk_start}-{chunk_end-1})")
            
            # Monitor memory usage
            memory_percent = psutil.virtual_memory().percent
            self.logger.info(f"Memory usage before chunk: {memory_percent:.1f}%")
            
            try:
                # Select chunk
                chunk_dataset = dataset.select(range(chunk_start, chunk_end))
                
                # Process chunk
                processed_chunk = chunk_dataset.map(
                    desc=f"Preprocessing {split_name} chunk {chunk_num}",
                    remove_columns=chunk_dataset.column_names,
                    **map_kwargs
                )
                
                # No need to filter - preprocessing only returns successful images
                processed_chunks.append(processed_chunk)
                
                # Save progress after each chunk
                if processed_chunks:
                    current_processed = concatenate_datasets(processed_chunks)
                    self._save_chunk_progress(current_processed, split_name, chunk_num)
                    
                    self.logger.info(f"Chunk {chunk_num} completed: {len(processed_chunk)} samples processed")
                    self.logger.info(f"Total processed so far: {len(current_processed)} samples")
                
                # Force garbage collection after each chunk
                gc.collect()
                
                # Monitor memory after processing
                memory_percent = psutil.virtual_memory().percent
                self.logger.info(f"Memory usage after chunk: {memory_percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_num}: {e}")
                self.logger.info("Saving progress before handling error...")
                
                # Save what we have so far
                if processed_chunks:
                    current_processed = concatenate_datasets(processed_chunks)
                    self._save_chunk_progress(current_processed, split_name, chunk_num - 1)
                
                # Re-raise the error
                raise e
        
        # Combine all chunks
        if processed_chunks:
            final_dataset = concatenate_datasets(processed_chunks)
            self.logger.info(f"Completed processing {split_name}: {len(final_dataset)} samples")
            return final_dataset
        else:
            self.logger.warning(f"No samples were successfully processed for {split_name}")
            return None

    def _save_chunk_progress(self, dataset, split_name, chunk_num):
        """Save progress after processing each chunk."""
        try:
            progress_dir = self.output_dir / f"{self.processor_name}_progress"
            progress_dir.mkdir(exist_ok=True)
            
            progress_file = progress_dir / f"{split_name}_chunk_{chunk_num}"
            dataset.save_to_disk(str(progress_file))
            
            self.logger.info(f"Saved progress for {split_name} chunk {chunk_num} to {progress_file}")
        except Exception as e:
            self.logger.warning(f"Could not save chunk progress: {e}")

    def _save_dataset(self) -> Dict[str, str]:
        """Save the processed dataset to disk using HuggingFace datasets."""
        saved_files = {}
        
        self.logger.info("Saving dataset using HuggingFace save_to_disk...")
        
        if isinstance(self.processed_dataset, DatasetDict):
            # Save entire DatasetDict to disk
            filename = f"{self.processor_name}"
            filepath = self.output_dir / filename
            self.processed_dataset.save_to_disk(str(filepath))
            saved_files["dataset_dict"] = str(filepath)
        else:
            # Save single dataset to disk
            filename = f"{self.processor_name}"
            filepath = self.output_dir / filename
            self.processed_dataset.save_to_disk(str(filepath))
            saved_files["dataset"] = str(filepath)
        
        self.logger.info(f"Saved dataset to: {filepath}")
        return saved_files

    def process(self) -> Dict[str, Any]:
        """
        Process the dataset and save to disk using HuggingFace datasets.
        
        Returns:
            Dictionary containing processing results, file paths, and metadata
        """
        self.logger.info(f"Starting dataset processing: {self.processor_name}")
        
        # Apply preprocessing
        self._apply_preprocessing()
        
        # Save dataset
        saved_files = self._save_dataset()
        
        # Compile results
        results = {
            "processor_name": self.processor_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir),
            "saved_files": saved_files,
            "split_limits": self.split_limits,
            "num_threads": self.num_threads,
            "success": True
        }
        
        self.logger.info("Dataset processing completed successfully")
        return results