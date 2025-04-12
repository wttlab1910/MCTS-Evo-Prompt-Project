# define task prompts for various datasets
from tasks.base_task import BaseDataset, BaseTask
import re
import string
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

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "TREC",
                 task_description = "",
                 data_dir='',  
                 seed=None, 
                 
                 post_instruction=True, 
                 TaskDataset=BaseDataset,
                 option_num=6, 
                 **kwargs):
        self.options = {
            'abbreviation':'A',
            'entity':'B',
            'description and abstract concept':'C',
            'human being':'D',
            'location':'E',
            'numeric value':'F'
        }
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        TaskDataset=TaskDataset,
                        option_num=option_num,
                        )
        self.answer_format_prompt = "Answer:"
        
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('trec')
        question_format = "Text: {text}\nAssign a label for the preceding text\nOptions:\n(A) Abbreviation\n(B) Entity\n(C) Description and abstract concept\n(D) Human being\n(E) Location\n(F) Numeric value"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            question_str = question_format.format(
                text=example['text'], 
                )
            new_dataset['train'].append(dict(question=question_str, answer=chr(65 + example['coarse_label'])))
        for example in dataset['test']:
            question_str = question_format.format(
                text=example['text'], 
                )
            new_dataset['test'].append(dict(question=question_str, answer=chr(65 + example['coarse_label'])))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        letters = string.ascii_uppercase[:len(self.options)] + string.ascii_lowercase[:len(self.options)]

        answer = re.findall(r"\([" + letters + r"]\)", response.lower())
        if len(answer)>0:
            return answer[-1][1].upper()
        
        pattern_str = '|'.join([re.escape(option) for option in self.options])
        backup_match = re.findall(pattern_str, response.lower(), re.IGNORECASE)

        if backup_match:
            return self.options[backup_match[-1].lower()]
        else:
            return 'N/A: Format error'
