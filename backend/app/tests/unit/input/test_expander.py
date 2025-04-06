"""
Unit tests for prompt expander component.
"""
import pytest
from app.core.input.prompt_expander import PromptExpander

class TestPromptExpander:
    """
    Tests for PromptExpander.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
        self.expander = PromptExpander()
    
    def test_prompt_expansion(self):
        """
        Test basic prompt expansion.
        """
        prompt = "Summarize this article."
        task_analysis = {
            "task_type": "summarization",
            "task_confidence": 0.9,
            "category": "text_summarization",
            "key_concepts": ["article", "summary"],
            "entities": [],
            "evaluation_methods": ["rouge", "semantic_similarity"],
            "prompt": prompt
        }
        
        expanded = self.expander.expand(prompt, task_analysis)
        
        assert "Role:" in expanded
        assert "Task:" in expanded
        assert "Steps:" in expanded
        assert "Output Format:" in expanded
    
    def test_domain_adaptation(self):
        """
        Test domain adaptation in expansion.
        """
        prompt = "Summarize this medical article."
        task_analysis = {
            "task_type": "summarization",
            "task_confidence": 0.9,
            "category": "text_summarization",
            "key_concepts": ["medical article", "summary"],
            "entities": [],
            "evaluation_methods": ["rouge", "semantic_similarity"],
            "prompt": prompt
        }
        
        expanded = self.expander.expand(prompt, task_analysis)
        
        assert "medical terminology" in expanded.lower()
    
    def test_classification_expansion(self):
        """
        Test expansion for classification tasks.
        """
        prompt = "Classify this review as positive or negative."
        task_analysis = {
            "task_type": "classification",
            "task_confidence": 0.9,
            "category": "sentiment_classification",
            "key_concepts": ["review", "sentiment"],
            "entities": [],
            "evaluation_methods": ["accuracy", "f1_score"],
            "prompt": prompt
        }
        
        expanded = self.expander.expand(prompt, task_analysis)
        
        assert "Classification" in expanded
        assert "Category:" in expanded
    
    def test_extraction_expansion(self):
        """
        Test expansion for extraction tasks.
        """
        prompt = "Extract the names of people mentioned in this text."
        task_analysis = {
            "task_type": "extraction",
            "task_confidence": 0.9,
            "category": "named_entity_recognition",
            "key_concepts": ["names", "people"],
            "entities": [],
            "evaluation_methods": ["exact_match", "f1_score"],
            "prompt": prompt
        }
        
        expanded = self.expander.expand(prompt, task_analysis)
        
        assert "Extraction" in expanded or "Information" in expanded
        assert "key-value pairs" in expanded.lower() or "structured format" in expanded.lower()
    
    def test_template_loading(self):
        """
        Test template loading functionality.
        """
        templates = self.expander.templates
        
        # Check if templates were loaded
        assert len(templates) > 0
        assert "classification" in templates
        assert "summarization" in templates
    
    def test_pattern_loading(self):
        """
        Test pattern loading functionality.
        """
        patterns = self.expander.patterns
        
        # Check if patterns were loaded
        assert len(patterns) > 0
        assert "classification" in patterns
        
        # Check pattern structure
        for task_type, task_patterns in patterns.items():
            if task_patterns:
                pattern = task_patterns[0]
                assert "pattern" in pattern
                assert "role" in pattern
                assert "steps" in pattern
                assert isinstance(pattern["steps"], list)
    
    def test_format_enhancement(self):
        """
        Test output format enhancement.
        """
        prompt = "Generate code to sort a list in Python."
        task_analysis = {
            "task_type": "code_generation",
            "task_confidence": 0.9,
            "category": "function_generation",
            "key_concepts": ["code", "python", "sort"],
            "entities": ["Python"],
            "evaluation_methods": ["functional_correctness", "code_quality"],
            "prompt": prompt
        }
        
        expanded = self.expander.expand(prompt, task_analysis)
        
        assert "Output Format:" in expanded
        assert "code" in expanded.lower() and "comments" in expanded.lower()