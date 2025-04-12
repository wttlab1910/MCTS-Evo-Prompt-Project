"""
Data loading utilities for MCTS-Evo-Prompt.
This module provides a local implementation of the load_dataset function
that task files use to load datasets from various sources.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

def load_dataset(dataset_name: str, subset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a dataset by name.
    
    This function provides an alternative to Hugging Face datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        subset: Optional subset name (specific to certain datasets)
        
    Returns:
        Dictionary containing the dataset
    """
    print(f"Loading dataset: {dataset_name}" + (f", subset: {subset}" if subset else ""))
    
    # Try to load from local files first
    local_dataset = _try_load_local_dataset(dataset_name, subset)
    if local_dataset is not None:
        return local_dataset
    
    # If not found, create a synthetic dataset
    return _create_synthetic_dataset(dataset_name, subset)

def _try_load_local_dataset(dataset_name: str, subset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Try to load dataset from local data directory."""
    # Normalize dataset name
    dataset_name = dataset_name.lower().replace("-", "_")
    
    # Common data file locations
    data_paths = [
        f"data/tasks/{dataset_name}.json",
        f"data/{dataset_name}.json",
        f"backend/data/tasks/{dataset_name}.json",
        f"backend/data/{dataset_name}.json", 
        f"app/data/tasks/{dataset_name}.json"
    ]
    
    # Check if any of the paths exist and load the data
    for path in data_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Loaded dataset {dataset_name} from {path}")
                
                # Handle subset if provided
                if subset and isinstance(data, dict) and subset in data:
                    return data[subset]
                    
                return data
            except Exception as e:
                print(f"Error loading dataset from {path}: {e}")
    
    return None

def _create_synthetic_dataset(dataset_name: str, subset: Optional[str] = None) -> Dict[str, Any]:
    """Create a minimal synthetic dataset for testing."""
    print(f"Creating synthetic dataset for {dataset_name}")
    
    # Task-specific synthetic data
    if dataset_name == "trec":
        dataset = {
            "train": [
                {"text": "What is the capital of France?", "coarse_label": 0},
                {"text": "Who invented the telephone?", "coarse_label": 3}
            ],
            "test": [
                {"text": "What is the tallest mountain?", "coarse_label": 0},
                {"text": "When was the Declaration of Independence signed?", "coarse_label": 5}
            ]
        }
    elif dataset_name == "super_glue" and subset == "cb":
        dataset = {
            "train": [
                {"premise": "The man is sleeping.", "hypothesis": "The man is awake.", "label": 1},
                {"premise": "The bird is flying.", "hypothesis": "The bird has wings.", "label": 0}
            ],
            "validation": [
                {"premise": "The cat is on the mat.", "hypothesis": "The mat is under the cat.", "label": 0},
                {"premise": "It's raining outside.", "hypothesis": "The ground is wet.", "label": 0}
            ]
        }
    elif dataset_name == "biosses":
        dataset = {
            "train": [
                {"sentence1": "Protein kinase C (PKC) is activated by diacylglycerol.", 
                 "sentence2": "PKC activation is carried out by diacylglycerol.", 
                 "score": 4.0},
                {"sentence1": "BNIP3 interacts with LC3 and this leads to mitophagy.", 
                 "sentence2": "BNIP3 causes internalisation of LC3 in the mitochondria.", 
                 "score": 3.0}
            ]
        }
    elif dataset_name == "ncbi_disease":
        dataset = {
            "train": [
                {"tokens": ["Mutation", "in", "the", "APC", "gene", "causes", "colorectal", "cancer", "."],
                 "ner_tags": [0, 0, 0, 1, 2, 0, 1, 2, 0]},
                {"tokens": ["The", "BRCA1", "gene", "is", "linked", "to", "breast", "cancer", "."],
                 "ner_tags": [0, 1, 2, 0, 0, 0, 1, 2, 0]}
            ],
            "validation": [
                {"tokens": ["Cystic", "fibrosis", "is", "caused", "by", "a", "mutation", "in", "the", "CFTR", "gene", "."],
                 "ner_tags": [1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0]}
            ],
            "test": [
                {"tokens": ["Huntington's", "disease", "is", "a", "neurodegenerative", "disorder", "."],
                 "ner_tags": [1, 2, 0, 0, 0, 0, 0]}
            ]
        }
    elif dataset_name == "SetFit/subj":
        dataset = {
            "train": [
                {"text": "This movie was fantastic and I loved every minute of it.", "label": 1, "label_text": "Subjective"},
                {"text": "The temperature today is 25 degrees Celsius.", "label": 0, "label_text": "Objective"}
            ],
            "test": [
                {"text": "I think the book was boring and too long.", "label": 1, "label_text": "Subjective"},
                {"text": "The Earth revolves around the Sun.", "label": 0, "label_text": "Objective"}
            ]
        }
    elif dataset_name == "bigbio/med_qa":
        dataset = {
            "train": [
                {"question": "Which of the following is a symptom of pneumonia?",
                 "options": [{"key": "A", "value": "Fever"}, {"key": "B", "value": "Hair loss"}, 
                             {"key": "C", "value": "Skin rash"}, {"key": "D", "value": "Joint pain"}],
                 "answer": "Fever", "answer_idx": 0},
                {"question": "What is the function of insulin?",
                 "options": [{"key": "A", "value": "Decrease blood glucose"}, {"key": "B", "value": "Increase blood pressure"}, 
                             {"key": "C", "value": "Decrease heart rate"}, {"key": "D", "value": "Increase body temperature"}],
                 "answer": "Decrease blood glucose", "answer_idx": 0}
            ],
            "test": [
                {"question": "Which organ is primarily responsible for filtering blood?",
                 "options": [{"key": "A", "value": "Kidney"}, {"key": "B", "value": "Liver"}, 
                             {"key": "C", "value": "Spleen"}, {"key": "D", "value": "Lung"}],
                 "answer": "Kidney", "answer_idx": 0}
            ]
        }
    else:
        # Create a minimal dataset structure for any other dataset
        dataset = {
            "train": [
                {"question": "Sample question 1", "answer": "Sample answer 1"},
                {"question": "Sample question 2", "answer": "Sample answer 2"}
            ],
            "test": [
                {"question": "Test question 1", "answer": "Test answer 1"},
                {"question": "Test question 2", "answer": "Test answer 2"}
            ]
        }
    
    # Create directory to save synthetic data for future use
    if dataset_name:
        try:
            data_dir = Path("data/tasks")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{dataset_name}.json"
            if subset:
                filename = f"{dataset_name}_{subset}.json"
            
            with open(data_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
                
            print(f"Saved synthetic dataset to {data_dir / filename}")
        except Exception as e:
            print(f"Error saving synthetic dataset: {e}")
    
    return dataset