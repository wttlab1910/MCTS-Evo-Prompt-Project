"""
Unit tests for prompt separator component.
"""
import pytest
from app.core.input.prompt_separator import PromptSeparator

class TestPromptSeparator:
    """
    Tests for PromptSeparator.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
        self.separator = PromptSeparator()
    
    def test_explicit_delimiter_separation(self):
        """
        Test separation with explicit delimiters.
        """
        input_text = "Instruction: Summarize the following text. Data: This is some example text to summarize."
        prompt, data = self.separator.separate(input_text)
        
        assert prompt == "Summarize the following text."
        assert data == "This is some example text to summarize."
    
    def test_another_explicit_delimiter(self):
        """
        Test separation with another explicit delimiter format.
        """
        input_text = "Prompt: Classify the sentiment. Content: I really enjoyed this product!"
        prompt, data = self.separator.separate(input_text)
        
        assert prompt == "Classify the sentiment."
        assert data == "I really enjoyed this product!"
    
    def test_semantic_boundary_separation(self):
        """
        Test separation with semantic boundaries.
        """
        input_text = "Please analyze the sentiment of the following review.\n\nThis product exceeded my expectations!"
        prompt, data = self.separator.separate(input_text)
        
        assert prompt == "Please analyze the sentiment of the following review."
        assert data == "This product exceeded my expectations!"
    
    def test_directive_based_separation(self):
        """
        Test separation based on directive phrases.
        """
        input_text = "Please analyze the following text: The company reported increased profits in Q3."
        prompt, data = self.separator.separate(input_text)
        
        # Check that the split happened correctly
        assert "Please analyze" in prompt
        assert "The company reported" in data
    
    def test_structured_format_separation(self):
        """
        Test separation with structured format (JSON-like).
        """
        input_text = '{"instruction": "Extract the main entities", "data": "Apple announced a new product yesterday."}'
        prompt, data = self.separator.separate(input_text)
        
        # The test might be less strict due to JSON parsing intricacies
        assert "Extract" in prompt
        assert "Apple" in data
    
    def test_ambiguous_case_handling(self):
        """
        Test handling of ambiguous cases.
        """
        input_text = "What is the main theme of this text? The novel explores themes of identity and belonging."
        prompt, data = self.separator.separate(input_text)
        
        assert len(prompt) > 0
        assert len(data) > 0
        
    def test_no_data_case(self):
        """
        Test case with only a prompt and no data.
        """
        input_text = "Summarize the following article."
        prompt, data = self.separator.separate(input_text)
        
        assert prompt == input_text
        assert data == ""