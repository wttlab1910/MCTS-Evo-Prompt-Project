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
                 
                 task_name = 'biosses', 
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
        self.option_dict = {'not similar':'A', 'somewhat similar':'B', 'similar':'C'}
        
    def discretize(self, score):
        if score < 5/3:
            return "A"
        elif score < 10/3:
            return "B"
        else:
            return "C"
            
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('biosses')
        question_format = "Sentence1: {sentence1}\nSentence2: {sentence2}\nOptions:\n(A) not similar\n(B) somewhat similar\n(C) similar"
        new_dataset = []
        for example in dataset['train']:
            question_str = question_format.format(
                sentence1=example['sentence1'], 
                sentence2 = example['sentence2'],
                )
            new_dataset.append(dict(question=question_str, answer=self.discretize(example['score'])))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\((A|B|C)\)"
        match = re.findall(clean_pattern, response)
        if len(match) != 0:
            return match[-1]
        clean_pattern = r"\b(not similar|somewhat similar|similar)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return self.option_dict[match[-1]]
    
        return "N/A: format error."
    