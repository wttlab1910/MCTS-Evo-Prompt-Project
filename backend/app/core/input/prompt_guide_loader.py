"""
Module for loading and managing Prompt Engineering Guide content.
"""
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.utils.logger import get_logger
from app.config import PROMPT_GUIDE_DIR

logger = get_logger("input.prompt_guide_loader")

class PromptGuideLoader:
    """
    Loads and manages Prompt Engineering Guide techniques, templates and examples.
    """
    
    def __init__(self):
        """
        Initialize the prompt guide loader.
        """
        self.techniques_dir = PROMPT_GUIDE_DIR / "techniques"
        self.templates_dir = PROMPT_GUIDE_DIR / "templates"
        self.examples_dir = PROMPT_GUIDE_DIR / "examples"
        
        # Create directories if they don't exist
        for directory in [self.techniques_dir, self.templates_dir, self.examples_dir]:
            directory.mkdir(exist_ok=True, parents=True)
            
        # Load techniques, templates and examples
        self.techniques = self._load_techniques()
        self.templates = self._load_templates()
        self.examples = self._load_examples()
        
        logger.info(f"Loaded {len(self.techniques)} techniques, {len(self.templates)} templates, and {len(self.examples)} example sets")
        
    def _load_json_files(self, directory: Path) -> Dict[str, Any]:
        """
        Load all JSON files from a directory.
        
        Args:
            directory: Directory containing JSON files.
            
        Returns:
            Dictionary mapping filenames (without extension) to file contents.
        """
        result = {}
        
        if not directory.exists():
            return result
            
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    result[file_path.stem] = data
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return result
        
    def _load_techniques(self) -> Dict[str, Any]:
        """
        Load prompting techniques.
        
        Returns:
            Dictionary of techniques.
        """
        return self._load_json_files(self.techniques_dir)
        
    def _load_templates(self) -> Dict[str, Any]:
        """
        Load task-specific templates.
        
        Returns:
            Dictionary of templates.
        """
        return self._load_json_files(self.templates_dir)
        
    def _load_examples(self) -> Dict[str, Any]:
        """
        Load examples.
        
        Returns:
            Dictionary of examples.
        """
        return self._load_json_files(self.examples_dir)
        
    def get_technique(self, technique_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific technique.
        
        Args:
            technique_name: Name of the technique.
            
        Returns:
            Technique data or None if not found.
        """
        return self.techniques.get(technique_name)
        
    def get_template(self, task_type: str) -> Optional[Dict[str, Any]]:
        """
        Get templates for a specific task type.
        
        Args:
            task_type: Type of task.
            
        Returns:
            Template data or None if not found.
        """
        return self.templates.get(task_type)
        
    def get_examples(self, task_type: str) -> Optional[Dict[str, Any]]:
        """
        Get examples for a specific task type.
        
        Args:
            task_type: Type of task.
            
        Returns:
            Examples data or None if not found.
        """
        return self.examples.get(f"{task_type}_examples")
        
    def get_all_techniques(self) -> List[str]:
        """
        Get all available technique names.
        
        Returns:
            List of technique names.
        """
        return list(self.techniques.keys())
        
    def get_all_task_types(self) -> List[str]:
        """
        Get all available task types with templates.
        
        Returns:
            List of task types.
        """
        return list(self.templates.keys())
        
    def generate_training_data(self) -> List[Dict[str, str]]:
        """
        Generate training data for prompt expansion model.
        
        Returns:
            List of training examples with 'input' and 'target' keys.
        """
        training_data = []
        
        # Generate examples from templates
        for task_type, template_data in self.templates.items():
            if "templates" not in template_data:
                continue
                
            for template_info in template_data["templates"]:
                if template_info.get("format") != "detailed":
                    continue
                    
                # Generate simple-to-detailed prompt pairs
                simple_prompt = f"Help me {task_type} this content"
                detailed_prompt = template_info["template"]
                
                # Replace placeholders with generic values
                detailed_prompt = detailed_prompt.replace("{categories}", "relevant categories")
                detailed_prompt = detailed_prompt.replace("{content}", "{content}")
                
                training_data.append({
                    "input": simple_prompt,
                    "target": detailed_prompt
                })
                
        # Generate examples from techniques
        for technique_name, technique_data in self.techniques.items():
            if "examples" not in technique_data:
                continue
                
            for example in technique_data["examples"]:
                if "prompt" not in example or "task" not in example:
                    continue
                    
                # Get the proper technique name from the data
                technique_display_name = technique_data.get("name", technique_name)
                
                # Create training example (use the actual technique name, not the filename)
                simple_prompt = f"Use {technique_display_name} for {example['task']}"
                target_prompt = example["prompt"]
                
                training_data.append({
                    "input": simple_prompt,
                    "target": target_prompt
                })
        
        # Add direct technique application examples
        for technique_name, technique_data in self.techniques.items():
            if "name" in technique_data:
                # Use the actual technique name from the data
                technique_display_name = technique_data.get("name", technique_name)
                simple_prompt = f"Apply {technique_display_name} technique"
                
                if "template" in technique_data:
                    target_prompt = technique_data["template"]
                    
                    # Replace placeholders with generic values
                    target_prompt = target_prompt.replace("{categories}", "relevant categories")
                    target_prompt = target_prompt.replace("{content}", "input content")
                    
                    training_data.append({
                        "input": simple_prompt,
                        "target": target_prompt
                    })
                
        return training_data
        
    def create_default_guide(self):
        """
        Create default Prompt Engineering Guide content if none exists.
        """
        # Check if files already exist
        if self.techniques or self.templates or self.examples:
            return
            
        # Create default zero-shot technique
        zero_shot = {
            "name": "Zero-Shot Prompting",
            "description": "Zero-shot prompting means instructing the model to perform a task without providing examples or demonstrations.",
            "key_concepts": [
                "No examples needed",
                "Direct instructions",
                "Clear task specification"
            ],
            "examples": [
                {
                    "task": "sentiment_classification",
                    "prompt": "Classify the text into neutral, negative or positive.\nText: I think the vacation is okay.\nSentiment:",
                    "explanation": "This prompt directly asks the model to classify without showing examples first"
                }
            ],
            "best_practices": [
                "Be clear and specific about the task",
                "Specify the expected output format", 
                "Include relevant constraints or considerations"
            ],
            "template": "Classify the following text as {categories}.\nText: {content}\nClassification:"
        }
        
        # Create default classification template
        classification = {
            "task_type": "classification",
            "templates": [
                {
                    "name": "Basic Classification",
                    "template": "Classify the following text into one of these categories: {categories}.\n\nText: {content}\n\nClassification:",
                    "format": "simple"
                },
                {
                    "name": "Expert Classification",
                    "template": "As a classification expert, analyze the following text and categorize it into one of these classes: {categories}.\n\nText: {content}\n\nFirst, identify key features in the text relevant to classification.\nThen, determine which category best matches these features.\n\nClassification:",
                    "format": "detailed"
                }
            ],
            "examples": [
                {
                    "input": "I really enjoyed this product. It exceeded my expectations!",
                    "categories": "positive, negative, neutral",
                    "expected_output": "positive",
                    "explanation": "The text expresses clear enjoyment and satisfaction"
                }
            ]
        }
        
        # Save default files
        with open(self.techniques_dir / "zero_shot.json", 'w', encoding='utf-8') as f:
            json.dump(zero_shot, f, indent=2)
            
        with open(self.templates_dir / "classification.json", 'w', encoding='utf-8') as f:
            json.dump(classification, f, indent=2)
            
        logger.info("Created default Prompt Engineering Guide content")