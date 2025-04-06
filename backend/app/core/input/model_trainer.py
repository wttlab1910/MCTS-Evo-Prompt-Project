"""
Module for training a prompt expansion model based on Prompt Engineering Guidelines.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from tqdm import tqdm
import random
from app.utils.logger import get_logger
from app.config import PROMPT_GUIDE_DIR, PROMPT_EXPANSION_MODEL_PATH

logger = get_logger("input.model_trainer")

class PromptDataset(Dataset):
    """
    Dataset for training prompt expansion model.
    """
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            example["input"], 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            example["target"], 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()
        
        # Replace padding token id with -100 so it's ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class PromptModelTrainer:
    """
    Trainer for prompt expansion model.
    """
    
    def __init__(self, model_name="google/flan-t5-small"):
        """
        Initialize the prompt model trainer.
        
        Args:
            model_name: Base model to fine-tune (default: flan-t5-small).
        """
        self.model_name = model_name
        self.prompt_guide_dir = PROMPT_GUIDE_DIR
        self.model_path = PROMPT_EXPANSION_MODEL_PATH
        
        # Create directories if they don't exist
        self.prompt_guide_dir.mkdir(exist_ok=True, parents=True)
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer initialized for {model_name}")
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise
    
    def load_or_create_training_data(self) -> List[Dict[str, str]]:
        """
        Load or create training data for prompt expansion.
        
        Returns:
            List of examples with 'input' and 'target' keys.
        """
        training_data_file = self.prompt_guide_dir / "training_data.json"
        
        # Return existing training data if available
        if training_data_file.exists():
            try:
                with open(training_data_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} training examples")
                return data
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
        
        # Create synthetic training data from Prompt Engineering Guide
        logger.info("Creating training data from Prompt Engineering Guide")
        
        # Load data from Prompt Guide
        from app.core.input.prompt_guide_loader import PromptGuideLoader
        guide_loader = PromptGuideLoader()
        
        # Create default guide content if none exists
        guide_loader.create_default_guide()
        
        # Generate training data from guide
        examples = guide_loader.generate_training_data()
        
        # If no examples were generated, fall back to synthetic generation
        if not examples:
            examples = self._generate_synthetic_examples(
                guide_loader.templates, 
                guide_loader.techniques
            )
        
        # Save training data
        try:
            with open(training_data_file, 'w') as f:
                json.dump(examples, f, indent=2)
            logger.info(f"Saved {len(examples)} training examples to {training_data_file}")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
        
        return examples
    
    def _generate_synthetic_examples(self, templates: Dict[str, Any], patterns: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """
        Generate synthetic training examples from templates and patterns.
        
        Args:
            templates: Dictionary of templates by task type.
            patterns: Dictionary of patterns by task type.
            
        Returns:
            List of examples with 'input' and 'target' keys.
        """
        examples = []
        
        # Simple prompts to use as inputs
        simple_prompts = {
            "classification": [
                "Classify this text.",
                "Categorize the following content.",
                "Which category does this belong to?",
                "Determine the type of the following.",
                "Is this positive or negative?",
                "Classify this email as spam or not."
            ],
            "extraction": [
                "Extract the key information.",
                "Find the main entities in this text.",
                "Extract the names and dates.",
                "What are the key points mentioned?",
                "Who is mentioned in this article?",
                "Extract the email addresses from this text."
            ],
            "summarization": [
                "Summarize this article.",
                "Give me a summary.",
                "What are the main points?",
                "Can you summarize this?",
                "Provide a brief summary of the following.",
                "TLDR of this document."
            ],
            "generation": [
                "Write a paragraph about climate change.",
                "Generate a product description.",
                "Create a story about space exploration.",
                "Write an email to a customer.",
                "Generate content for my website.",
                "Write a blog post about AI."
            ],
            "question_answering": [
                "What causes climate change?",
                "How does photosynthesis work?",
                "Why is the sky blue?",
                "What is the capital of France?",
                "How do I reset my password?",
                "When was the first computer invented?"
            ],
            "sentiment_analysis": [
                "Determine the sentiment of this review.",
                "Is this feedback positive or negative?",
                "Analyze the emotional tone of this text.",
                "What is the sentiment expressed in this comment?",
                "How does the author feel about this product?",
                "Identify the sentiment in this social media post."
            ],
            "code_generation": [
                "Write a function to sort an array.",
                "Create a Python class for a bank account.",
                "Generate code to parse JSON data.",
                "Implement a binary search algorithm.",
                "Code a simple REST API endpoint.",
                "Write a JavaScript function to validate an email."
            ]
        }
        
        # Process template formats
        processed_templates = {}
        for task_type, template_data in templates.items():
            if isinstance(template_data, dict) and "templates" in template_data:
                # New format with multiple templates
                for template_info in template_data["templates"]:
                    if template_info.get("format") == "detailed":
                        processed_templates[task_type] = template_info["template"]
                        break
                # If no detailed template found, use the first one
                if task_type not in processed_templates and template_data["templates"]:
                    processed_templates[task_type] = template_data["templates"][0]["template"]
            else:
                # Old format with single template
                processed_templates[task_type] = template_data
        
        # Generate examples for each task type
        for task_type, template in processed_templates.items():
            if task_type == "default":
                continue
                
            # Get patterns for this task type
            task_patterns = patterns.get(task_type, [])
            if not task_patterns:
                continue
                
            # Get simple prompts for this task type
            task_prompts = simple_prompts.get(task_type, ["Process this information."])
            
            # Generate examples for each pattern and prompt
            for pattern in task_patterns:
                role = pattern.get("role", "Task Specialist")
                steps = pattern.get("steps", ["Analyze", "Process", "Generate"])
                
                # Format steps as bullet points
                steps_text = "\n".join([f"- {step}" for step in steps])
                
                for prompt in task_prompts:
                    # Create expanded version using template
                    try:
                        expanded = template.format(
                            role=role,
                            steps=steps_text,
                            content=prompt,
                            categories="relevant categories"  # Add default for classification
                        )
                    except KeyError as e:
                        # Skip if template format doesn't match
                        logger.warning(f"Template format error for {task_type}: {e}")
                        continue
                    
                    # Add example
                    examples.append({
                        "input": prompt,
                        "target": expanded
                    })
        
        # Add task-agnostic examples
        default_template = processed_templates.get("default", "")
        if default_template:
            general_prompts = [
                "Help me with this task.",
                "I need assistance with the following.",
                "Can you process this for me?",
                "Analyze this information.",
                "Complete this task."
            ]
            
            for prompt in general_prompts:
                try:
                    expanded = default_template.format(
                        role="Task Specialist",
                        steps="- Understand the requirements\n- Process the information\n- Generate appropriate response",
                        content=prompt
                    )
                except KeyError:
                    continue
                
                examples.append({
                    "input": prompt,
                    "target": expanded
                })
        
        # Shuffle examples
        random.shuffle(examples)
        
        return examples
    
    def train(self, epochs=3, batch_size=8, learning_rate=5e-5):
        """
        Train the prompt expansion model.
        
        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for training.
            
        Returns:
            Trained model path.
        """
        # Load or create training data
        examples = self.load_or_create_training_data()
        
        if not examples:
            logger.error("No training examples available")
            return None
        
        # Split into train and validation sets
        train_size = int(0.9 * len(examples))
        train_examples = examples[:train_size]
        val_examples = examples[train_size:]
        
        logger.info(f"Training with {len(train_examples)} examples, validating with {len(val_examples)} examples")
        
        # Initialize model
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info(f"Model initialized from {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return None
        
        # Create datasets
        train_dataset = PromptDataset(train_examples, self.tokenizer)
        val_dataset = PromptDataset(val_examples, self.tokenizer)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.model_path / "logs"),
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True
        )
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train model
        logger.info("Starting model training")
        try:
            trainer.train()
            
            # Save model
            trainer.save_model(str(self.model_path))
            self.tokenizer.save_pretrained(str(self.model_path))
            
            logger.info(f"Model trained and saved to {self.model_path}")
            return self.model_path
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return None
    
    def expand_prompt(self, prompt: str) -> str:
        """
        Expand a prompt using the trained model.
        
        Args:
            prompt: Input prompt text.
            
        Returns:
            Expanded prompt.
        """
        # Check if model exists
        if not self.model_path.exists():
            logger.warning("Model doesn't exist, training first")
            self.train()
        
        try:
            # Load model and tokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate expanded prompt
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode output
            expanded_prompt = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            return expanded_prompt
        except Exception as e:
            logger.error(f"Error expanding prompt with model: {e}")
            
            # Fallback to rule-based expansion
            from app.core.input.task_analyzer import TaskAnalyzer
            from app.core.input.prompt_expander import PromptExpander
            
            analyzer = TaskAnalyzer()
            expander = PromptExpander()
            
            task_analysis = analyzer.analyze(prompt)
            task_analysis["prompt"] = prompt
            expanded_prompt = expander.expand(prompt, task_analysis)
            
            return expanded_prompt