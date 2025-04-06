"""
Unit tests for prompt model trainer component.
"""
import pytest
import os
from pathlib import Path
from app.core.input.model_trainer import PromptModelTrainer, PromptDataset
from app.config import PROMPT_EXPANSION_MODEL_PATH

class TestPromptModelTrainer:
    """
    Tests for PromptModelTrainer.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
        self.trainer = PromptModelTrainer(model_name="google/flan-t5-small")
    
    def test_load_or_create_training_data(self):
        """
        Test loading or creating training data.
        """
        examples = self.trainer.load_or_create_training_data()
        
        # Check if training data was created
        assert isinstance(examples, list)
        assert len(examples) > 0
        assert all("input" in ex and "target" in ex for ex in examples)
    
    def test_synthetic_example_generation(self):
        """
        Test generation of synthetic examples.
        """
        # Create minimal templates and patterns
        templates = {
            "classification": "Role: {role}\nTask: Classify\nSteps: {steps}\nContent: {content}",
            "default": "Role: {role}\nTask: {content}\nSteps: {steps}"
        }
        
        patterns = {
            "classification": [
                {"pattern": "classify", "role": "Classifier", "steps": ["Read", "Classify"]}
            ]
        }
        
        examples = self.trainer._generate_synthetic_examples(templates, patterns)
        
        # Check if examples were generated
        assert len(examples) > 0
        assert all(isinstance(ex, dict) for ex in examples)
        assert all("input" in ex and "target" in ex for ex in examples)
    
    @pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="Skipping slow test")
    def test_model_training(self):
        """
        Test model training (slow, only run when explicitly enabled).
        """
        # Only run a quick training (1 epoch, small batch)
        result = self.trainer.train(epochs=1, batch_size=2)
        
        # Check if model was trained
        assert result is not None
        assert Path(result).exists()
    
    def test_prompt_dataset(self):
        """
        Test prompt dataset creation.
        """
        examples = [
            {"input": "Summarize this", "target": "Role: Summarizer\nTask: Summarize\nSteps: Read, Summarize"},
            {"input": "Classify this", "target": "Role: Classifier\nTask: Classify\nSteps: Read, Classify"}
        ]
        
        dataset = PromptDataset(examples, self.trainer.tokenizer, max_length=128)
        
        # Check dataset properties
        assert len(dataset) == 2
        
        # Check item format
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
    
    def test_fallback_expansion(self):
        """
        Test fallback expansion when model doesn't exist.
        """
        # Ensure the model doesn't exist for this test
        original_path = self.trainer.model_path
        self.trainer.model_path = Path("/nonexistent/path")
        
        prompt = "Summarize this article."
        expanded = self.trainer.expand_prompt(prompt)
        
        # Restore original path
        self.trainer.model_path = original_path
        
        # Check if fallback expansion worked
        assert expanded is not None
        assert len(expanded) > len(prompt)