"""
Train the prompt expansion model using prompt engineering guidelines.
"""
import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from data import Dataset

class PromptModelTrainer:
    """Trainer for the prompt expansion model."""
    
    def __init__(
        self,
        knowledge_base_dir="app/data/knowledge_base/prompt_guide",
        model_name="google/flan-t5-small",
        output_dir="app/data/models/prompt_expansion",
        max_epochs=3
    ):
        """
        Initialize the prompt model trainer.
        
        Args:
            knowledge_base_dir: Directory containing prompt guidelines.
            model_name: Base model to fine-tune.
            output_dir: Directory to save the model.
            max_epochs: Maximum number of training epochs.
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        print(f"Initialized PromptModelTrainer with base model: {model_name}")
    
    def load_knowledge_base(self):
        """
        Load the prompt engineering knowledge base.
        
        Returns:
            Dictionary containing techniques, templates, and examples.
        """
        knowledge_base = {
            "techniques": [],
            "templates": [],
            "examples": []
        }
        
        # Load techniques
        techniques_dir = self.knowledge_base_dir / "techniques"
        if techniques_dir.exists():
            for file_path in techniques_dir.glob("*.json"):
                with open(file_path, "r") as f:
                    technique = json.load(f)
                    knowledge_base["techniques"].append(technique)
        
        # Load templates
        templates_dir = self.knowledge_base_dir / "templates"
        if templates_dir.exists():
            for file_path in templates_dir.glob("*.json"):
                with open(file_path, "r") as f:
                    template = json.load(f)
                    knowledge_base["templates"].append(template)
        
        # Load examples
        examples_dir = self.knowledge_base_dir / "examples"
        if examples_dir.exists():
            for file_path in examples_dir.glob("*.json"):
                with open(file_path, "r") as f:
                    example = json.load(f)
                    knowledge_base["examples"].append(example)
        
        print(f"Loaded knowledge base with {len(knowledge_base['techniques'])} techniques, "
              f"{len(knowledge_base['templates'])} templates, "
              f"{len(knowledge_base['examples'])} example sets")
        
        return knowledge_base
    
    def generate_training_data(self, knowledge_base):
        """
        Generate training data from the knowledge base.
        
        Args:
            knowledge_base: Knowledge base dictionary.
            
        Returns:
            List of input-target pairs.
        """
        training_data = []
        
        # Generate examples from techniques
        for technique in knowledge_base["techniques"]:
            if "examples" in technique:
                for example in technique["examples"]:
                    if "task" in example and "prompt" in example:
                        # Create simple-to-detailed prompt pair
                        training_data.append({
                            "input": f"Help me with {example['task']}",
                            "target": example["prompt"]
                        })
        
        # Generate examples from templates
        for template in knowledge_base["templates"]:
            if "templates" in template:
                for template_item in template["templates"]:
                    if "task_type" in template_item and "template" in template_item:
                        # Create task-based prompt pairs
                        training_data.append({
                            "input": f"Create a prompt for {template_item['task_type']}",
                            "target": template_item["template"]
                        })
        
        # Generate examples from example sets
        for example_set in knowledge_base["examples"]:
            if "task" in example_set and "examples" in example_set:
                for i, example in enumerate(example_set["examples"]):
                    if i == 0 and "prompt" in example:  # Use first example as simple prompt
                        # Create original-to-enhanced pairs
                        if len(example_set["examples"]) > 1 and "prompt" in example_set["examples"][1]:
                            training_data.append({
                                "input": example["prompt"],
                                "target": example_set["examples"][1]["prompt"]
                            })
        
        print(f"Generated {len(training_data)} training examples")
        return training_data
    
    def prepare_dataset(self, training_data):
        """
        Prepare the dataset for training.
        
        Args:
            training_data: List of input-target pairs.
            
        Returns:
            Training and validation datasets.
        """
        # Convert to datasets
        dataset = Dataset.from_dict({
            "input": [item["input"] for item in training_data],
            "target": [item["target"] for item in training_data],
        })
        
        # Split into training and validation
        dataset = dataset.train_test_split(test_size=0.1)
        
        # Tokenize function
        def tokenize_function(examples):
            inputs = self.tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)
            outputs = self.tokenizer(examples["target"], padding="max_length", truncation=True, max_length=512)
            
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": outputs.input_ids,
            }
        
        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        print(f"Prepared dataset with {len(tokenized_dataset['train'])} training examples "
              f"and {len(tokenized_dataset['test'])} validation examples")
        
        return tokenized_dataset
    
    def train(self):
        """Train the prompt expansion model."""
        # Load knowledge base
        knowledge_base = self.load_knowledge_base()
        
        # Generate training data
        training_data = self.generate_training_data(knowledge_base)
        
        if not training_data:
            print("No training data generated. Please check the knowledge base.")
            return
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset(training_data)
        
        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.max_epochs,
            predict_with_generate=True,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            report_to=["tensorboard"],
        )
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train model
        print("Starting model training...")
        trainer.train()
        
        # Save model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        print("Model training complete!")
    
    def test_model(self, test_prompts=None):
        """
        Test the trained model.
        
        Args:
            test_prompts: List of test prompts. If None, default examples are used.
        """
        if test_prompts is None:
            test_prompts = [
                "Classify the sentiment of this text.",
                "Summarize this article.",
                "Extract entities from this text.",
                "Write a story about space exploration."
            ]
        
        print("\nTesting model with example prompts:")
        for prompt in test_prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            
            # Generate expanded prompt
            outputs = self.model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            expanded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("\nInput prompt:")
            print(prompt)
            print("\nExpanded prompt:")
            print(expanded)
            print("-" * 50)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train the prompt expansion model")
    parser.add_argument("--knowledge-dir", default="app/data/knowledge_base/prompt_guide", 
                        help="Directory containing prompt guidelines")
    parser.add_argument("--model", default="google/flan-t5-small", 
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", default="app/data/models/prompt_expansion", 
                        help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--test-only", action="store_true", 
                        help="Only test the model, don't train")
    
    args = parser.parse_args()
    
    trainer = PromptModelTrainer(
        knowledge_base_dir=args.knowledge_dir,
        model_name=args.model,
        output_dir=args.output_dir,
        max_epochs=args.epochs
    )
    
    if not args.test_only:
        trainer.train()
    
    trainer.test_model()

if __name__ == "__main__":
    main()