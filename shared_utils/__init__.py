"""
Shared utilities for model training across different projects.

This package provides common training infrastructure that can be reused
across different model implementations.
"""

from .trainer import ModelTrainer
from .dataset_processor import DatasetProcessor
from .checkpoint_utils import (
    find_latest_checkpoint,
)

__version__ = "1.0.0"
__all__ = [
    "ModelTrainer", 
    "DatasetProcessor",
    "find_latest_checkpoint",
] 