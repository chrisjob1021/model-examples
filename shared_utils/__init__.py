"""
Shared utilities for model training across different projects.

This package provides common training infrastructure that can be reused
across different model implementations.
"""

from .trainer import ModelTrainer
from .dataset_processor import DatasetProcessor

__version__ = "1.0.0"
__all__ = ["ModelTrainer", "DatasetProcessor"] 