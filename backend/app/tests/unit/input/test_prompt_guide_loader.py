"""
Unit tests for prompt guide loader component.
"""
import pytest
import os
import json
import shutil
from pathlib import Path
from app.core.input.prompt_guide_loader import PromptGuideLoader
from app.config import PROMPT_GUIDE_DIR

class TestPromptGuideLoader:
    """
    Tests for PromptGuideLoader.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
        # Create a test directory for prompt guides
        self.test_guide_dir = PROMPT_GUIDE_DIR / "test"
        self.test_techniques_dir = self.test_guide_dir / "techniques"
        self.test_templates_dir = self.test_guide_dir / "templates"
        self.test_examples_dir = self.test_guide_dir / "examples"
        
        # Create test directories
        for directory in [self.test_techniques_dir, self.test_templates_dir, self.test_examples_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Create test loader
        self.loader = PromptGuideLoader()
    
    def teardown_method(self):
        """
        Clean up after tests.
        """
        # Remove test directory
        if self.test_guide_dir.exists():
            shutil.rmtree(self.test_guide_dir)
    
    def test_create_default_guide(self):
        """
        Test creating default guide.
        """
        # Create default guide
        self.loader.create_default_guide()
        
        # Check if files were created
        assert (self.loader.techniques_dir / "zero_shot.json").exists()
        assert (self.loader.templates_dir / "classification.json").exists()
    
    def test_load_techniques(self):
        """
        Test loading techniques.
        """
        # Create a test technique file
        test_technique = {
            "name": "Chain of Thought",
            "description": "Ask the model to show its reasoning step by step."
        }
        
        with open(self.test_techniques_dir / "chain_of_thought.json", 'w') as f:
            json.dump(test_technique, f)
        
        # Create a custom loader that uses the test directory
        class TestLoader(PromptGuideLoader):
            def __init__(self, test_dir):
                self.techniques_dir = test_dir / "techniques"
                self.templates_dir = test_dir / "templates"
                self.examples_dir = test_dir / "examples"
                self.techniques = self._load_techniques()
                self.templates = self._load_templates()
                self.examples = self._load_examples()
        
        # Load techniques
        test_loader = TestLoader(self.test_guide_dir)
        
        # Check if technique was loaded
        assert "chain_of_thought" in test_loader.techniques
        assert test_loader.techniques["chain_of_thought"]["name"] == "Chain of Thought"
    
    def test_generate_training_data(self):
        """
        Test generating training data.
        """
        # Create test files
        test_template = {
            "task_type": "summarization",
            "templates": [
                {
                    "name": "Detailed Summarization",
                    "template": "Summarize the following text in detail: {content}",
                    "format": "detailed"
                }
            ]
        }
        
        test_technique = {
            "name": "Few-Shot",
            "examples": [
                {
                    "task": "classification",
                    "prompt": "Example prompt for few-shot classification"
                }
            ]
        }
        
        # Write test files
        with open(self.test_templates_dir / "summarization.json", 'w') as f:
            json.dump(test_template, f)
            
        with open(self.test_techniques_dir / "few_shot.json", 'w') as f:
            json.dump(test_technique, f)
        
        # Create a custom loader that uses the test directory
        class TestLoader(PromptGuideLoader):
            def __init__(self, test_dir):
                self.techniques_dir = test_dir / "techniques"
                self.templates_dir = test_dir / "templates"
                self.examples_dir = test_dir / "examples"
                self.techniques = self._load_techniques()
                self.templates = self._load_templates()
                self.examples = self._load_examples()
        
        # Generate training data
        test_loader = TestLoader(self.test_guide_dir)
        training_data = test_loader.generate_training_data()
        
        # Print training data for debugging
        print("\nGenerated training data:")
        for example in training_data:
            print(f"Input: {example['input']}")
            print(f"Target: {example['target'][:50]}...\n")
        
        # Check if training data was generated
        assert len(training_data) > 0
        # Check if there's any example related to summarization
        assert any("summarization" in example["input"].lower() for example in training_data)
        # Check if there's any example related to Few-Shot technique
        assert any("few-shot" in example["input"].lower() for example in training_data)
    
    def test_get_technique(self):
        """
        Test retrieving a specific technique.
        """
        # Create a test technique
        self.loader.create_default_guide()
        
        # Get technique
        technique = self.loader.get_technique("zero_shot")
        
        # Check technique
        assert technique is not None
        assert technique["name"] == "Zero-Shot Prompting"
    
    def test_get_template(self):
        """
        Test retrieving a specific template.
        """
        # Create a test template
        self.loader.create_default_guide()
        
        # Get template
        template = self.loader.get_template("classification")
        
        # Check template
        assert template is not None
        assert template["task_type"] == "classification"
        assert len(template["templates"]) > 0
    
    def test_get_all_techniques(self):
        """
        Test retrieving all technique names.
        """
        # Create default techniques
        self.loader.create_default_guide()
        
        # Get all techniques
        techniques = self.loader.get_all_techniques()
        
        # Check techniques
        assert "zero_shot" in techniques
    
    def test_get_all_task_types(self):
        """
        Test retrieving all task types.
        """
        # Create default templates
        self.loader.create_default_guide()
        
        # Get all task types
        task_types = self.loader.get_all_task_types()
        
        # Check task types
        assert "classification" in task_types