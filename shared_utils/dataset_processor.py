from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import logging
import os

from datasets import DatasetDict, load_dataset

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
            **load_dataset_kwargs: Additional arguments to pass to load_dataset
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.processor_name = processor_name or "dataset_processor"
        self.preprocess_fn = preprocess_fn
        self.split_limits = split_limits or {}
        self.num_threads = num_threads or os.cpu_count()
        self.start_index = start_index
        self.load_dataset_kwargs = load_dataset_kwargs
        
        # Load the dataset using load_dataset
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Loading dataset: {dataset_name}")
        self.logger.info(f"Using {self.num_threads} threads for processing")
        
        self.dataset = load_dataset(dataset_name, **load_dataset_kwargs)
        
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
            'num_proc': 2,
            'load_from_cache_file': False,
            # 'writer_batch_size': 1000,
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
                
                processed_split = split_dataset.map(
                    desc=f"Preprocessing {split_name}",
                    remove_columns=split_dataset.column_names,
                    **map_kwargs
                )

                # Filter out failed images
                processed_split = processed_split.filter(lambda x: x['ok_flags'])

                self.processed_dataset[split_name] = processed_split
        else:
            # Process single dataset
            
            self.processed_dataset = self.dataset.map(
                desc="Preprocessing dataset",
                remove_columns=self.dataset.column_names,
                **map_kwargs
            )
            self.processed_dataset = self.processed_dataset.filter(lambda x: x['ok_flags'])

        
        self.logger.info("Preprocessing completed successfully")

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