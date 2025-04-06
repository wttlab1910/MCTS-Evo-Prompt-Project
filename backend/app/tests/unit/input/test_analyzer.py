"""
Unit tests for task analyzer component.
"""
import pytest
from app.core.input.task_analyzer import TaskAnalyzer

class TestTaskAnalyzer:
    """
    Tests for TaskAnalyzer.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
        self.analyzer = TaskAnalyzer()
    
    def test_classification_task_identification(self):
        """
        Test identification of classification tasks.
        """
        prompt = "Classify this text into one of the following categories: positive, negative, or neutral."
        analysis = self.analyzer.analyze(prompt)
        
        assert analysis["task_type"] == "classification"
        assert analysis["task_confidence"] > 0.5
    
    def test_extraction_task_identification(self):
        """
        Test identification of extraction tasks.
        """
        prompt = "Extract the main entities from the following text."
        analysis = self.analyzer.analyze(prompt)
        
        assert analysis["task_type"] == "extraction"
        assert analysis["task_confidence"] > 0.5
    
    def test_summarization_task_identification(self):
        """
        Test identification of summarization tasks.
        """
        prompt = "Summarize the following article in one paragraph."
        analysis = self.analyzer.analyze(prompt)
        
        assert analysis["task_type"] == "summarization"
        assert analysis["task_confidence"] > 0.5
    
    def test_generation_task_identification(self):
        """
        Test identification of generation tasks.
        """
        prompt = "Write a short story about a robot discovering emotions."
        analysis = self.analyzer.analyze(prompt)
        
        assert analysis["task_type"] == "generation"
        assert analysis["task_confidence"] > 0.5
    
    def test_question_answering_task_identification(self):
        """
        Test identification of question answering tasks.
        """
        prompt = "What are the main causes of climate change?"
        analysis = self.analyzer.analyze(prompt)
        
        assert analysis["task_type"] == "question_answering"
        assert analysis["task_confidence"] > 0.5
    
    def test_concept_extraction(self):
        """
        Test extraction of key concepts.
        """
        prompt = "Write an article about climate change and its impact on agriculture."
        analysis = self.analyzer.analyze(prompt)
        
        # Check if relevant concepts were extracted
        assert len(analysis["key_concepts"]) > 0
        assert any("climate" in concept.lower() for concept in analysis["key_concepts"]) or \
               any("agriculture" in concept.lower() for concept in analysis["key_concepts"])
    
    def test_entity_extraction(self):
        """
        Test extraction of entities.
        """
        prompt = "Analyze the speech given by Barack Obama on climate policy."
        analysis = self.analyzer.analyze(prompt)
        
        # Check if the named entity was extracted
        assert "Barack Obama" in analysis["entities"]
    
    def test_category_mapping(self):
        """
        Test mapping to predefined categories.
        """
        prompt = "Summarize this scientific article into three paragraphs."
        analysis = self.analyzer.analyze(prompt)
        
        # Check if mapped to a specific summarization category
        assert analysis["category"] in ["text_summarization", "abstractive_summarization", "extractive_summarization"]
    
    def test_evaluation_method_selection(self):
        """
        Test selection of evaluation methods.
        """
        prompt = "Translate this paragraph into French."
        analysis = self.analyzer.analyze(prompt)
        
        # Check if appropriate evaluation methods were selected
        assert "bleu" in analysis["evaluation_methods"] or "semantic_similarity" in analysis["evaluation_methods"]
    
    def test_default_task_fallback(self):
        """
        Test fallback to default task for ambiguous prompts.
        """
        prompt = "Process this information."  # Ambiguous
        analysis = self.analyzer.analyze(prompt)
        
        # Should default to "generation" for ambiguous prompts
        assert analysis["task_type"] == "generation"
        assert "task_confidence" in analysis