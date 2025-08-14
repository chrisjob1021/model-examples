"""
Checkpoint utilities for finding and managing training checkpoints.

This module provides utilities for finding the most recent checkpoint
in a directory, validating checkpoint integrity, and extracting
checkpoint metadata.
"""

import os
import json
from typing import Optional, Dict, Any, List


def find_latest_checkpoint(results_dir: str, checkpoint_prefix: str = "checkpoint-") -> Optional[str]:
    """
    Find the latest checkpoint in a results directory.
    
    Args:
        results_dir (str): Path to the directory containing checkpoints
        checkpoint_prefix (str): Prefix for checkpoint directories (default: "checkpoint-")
        
    Returns:
        str or None: Path to the latest checkpoint directory, or None if no checkpoints found
        
    Examples:
        >>> # Find latest checkpoint in results directory
        >>> latest = find_latest_checkpoint("/path/to/results/cnn_results_relu")
        >>> if latest:
        ...     print(f"Found checkpoint: {latest}")
        
        >>> # Find latest checkpoint with custom prefix
        >>> latest = find_latest_checkpoint("/path/to/results", "model_checkpoint-")
    """
    if not os.path.exists(results_dir):
        return None
    
    try:
        # Find all checkpoint directories
        checkpoint_dirs = [
            d for d in os.listdir(results_dir) 
            if d.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(results_dir, d))
        ]
        
        if not checkpoint_dirs:
            return None
        
        # Sort by checkpoint number (extract the number after the prefix)
        def extract_checkpoint_number(checkpoint_name: str) -> int:
            try:
                return int(checkpoint_name.split("-")[1])
            except (IndexError, ValueError):
                return -1
        
        checkpoint_dirs.sort(key=extract_checkpoint_number)
        latest_checkpoint = checkpoint_dirs[-1]
        
        checkpoint_path = os.path.join(results_dir, latest_checkpoint)
        return checkpoint_path
        
    except Exception as e:
        print(f"❌ Error finding checkpoints in {results_dir}: {e}")
        return None
