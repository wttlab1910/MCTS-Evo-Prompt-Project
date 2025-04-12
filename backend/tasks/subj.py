# define task prompts for various datasets
import re
try:
    # Try to import from huggingface directly
    from data import load_dataset as hf_load_dataset
    def load_dataset(name):
        return hf_load_dataset(name)
except ImportError:
    # Fallback to our local implementation
    try:
        from data.load_dataset import load_dataset
    except ImportError:
        try:
            from data import load_dataset
        except ImportError:
            # Create minimal implementation as last resort
            import os
            import json
            from pathlib import Path
            
            def load_dataset(name):
                """Simple fallback implementation for load_dataset"""
                print(f"Using minimal fallback load_dataset for {name}")
                
                # Create a simple dataset structure based on task name
                if name == "ncbi_disease":
                    return {
                        "train": [
                            {"tokens": ["Mutation", "in", "APC", "gene", "causes", "cancer"], 
                             "ner_tags": [0, 0, 1, 2, 0, 1]}
                        ],
                        "test": [
                            {"tokens": ["BRCA1", "mutation", "in", "breast", "cancer"],
                             "ner_tags": [1, 0, 0, 1, 2]}
                        ]
                    }
                # Add more cases for other datasets as needed
                
                return {"train": [], "test": []}
from tasks.base_task import BaseDataset, BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'subjective', 
                 task_discription = "",
                 seed=None, 
                 
                 post_instruction=True, 
                 **kwargs):
        super().__init__(task_name=task_name,
                         task_discription=task_discription,
                         seed=seed, 
                         train_size=train_size, 
                         eval_size=eval_size,
                         test_size = test_size, 
                         post_instruction=post_instruction,
                         )

        self.answer_format_prompt = ''
    
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('SetFit/subj')
        question_format = "Text: {text}\nIs the preceding text objective or subjective?\nOptions:\n- Objective\n- Subjective"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            question_str = question_format.format(
                text=example['text'], 
                )
            new_dataset['train'].append(dict(question=question_str, answer=example['label_text']))
        for example in dataset['test']:
            question_str = question_format.format(
                text=example['text'], 
                )
            new_dataset['test'].append(dict(question=question_str, answer=example['label_text']))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\b(objective|subjective)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    