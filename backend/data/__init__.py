"""
Data module for MCTS-Evo-Prompt.
"""
# Import and expose the load_dataset function
try:
    from .load_dataset import load_dataset
except ImportError:
    # Fallback implementation if the module isn't available
    import os
    import json
    from pathlib import Path
    
    def load_dataset(dataset_name, subset=None):
        """Simple fallback implementation for load_dataset"""
        print(f"Using fallback load_dataset for {dataset_name}")
        
        # Create a simple dataset structure
        return {
            "train": [],
            "test": []
        }