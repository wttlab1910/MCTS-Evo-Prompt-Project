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
                 test_size=None,  
                 
                 task_name = "epistemic",
                 task_description = "task from bigbench",
                 data_dir='',  
                 seed=None, 
                 
                 post_instruction=True, 
                 **kwargs):
        self.options = {}
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        )

        self.answer_format_prompt = "\nA:"
    
    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        json_data = self._load_json_file(data_dir)
        self.task_description = json_data['description']
        return json_data
    
    def transform_format(self, data):
        original_examples = data['examples']
        examples = []
        # Extracting input and target scores
        for example in original_examples:
            question = example['input']
            target_scores = example['target_scores']
            
            # Generating options and answer
            options = list(target_scores.keys())
            
            answer = [option.lower() for i, option in enumerate(options) if target_scores[option] == 1][0]

            options_str = 'Options:\n- entailment\n- non-entailment'
            question_str = "Identify the relation between the following premises and hypotheses, choosing from the options 'entailment' or 'non-entailment'.\n"+question+"\n"+options_str+'\n'
            
            # Formatting the output
            formatted_example = {
                'question': question_str,
                'answer': answer
            }
            examples.append(formatted_example)
        
        return examples
    
    def clean_response(self, response):
        clean_pattern = r"\b(entailment|non-entailment)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    