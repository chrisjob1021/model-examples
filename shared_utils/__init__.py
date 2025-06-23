"""
Shared utilities for model training across different projects.

This package provides common training infrastructure that can be reused
across different model implementations.
"""

from .trainer import ModelTrainer

__version__ = "1.0.0"
__all__ = ["ModelTrainer"] 