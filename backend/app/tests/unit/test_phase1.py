"""
Comprehensive tests for Phase 1: Input Processing and Initialization.
"""
import pytest
from app.core.input.prompt_separator import PromptSeparator
from app.core.input.task_analyzer import TaskAnalyzer
from app.core.input.prompt_expander import PromptExpander
from app.core.input.model_trainer import PromptModelTrainer
from app.services.prompt_service import PromptService

class TestPhase1:
    """
    Tests for Phase 1 components.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
        self.separator = PromptSeparator()
        self.analyzer = TaskAnalyzer()
        self.expander = PromptExpander()
        self.service = PromptService()
    
    def test_end_to_end_prompt_processing(self):
        """
        Test end-to-end prompt processing workflow.
        """
        # Test inputs
        test_inputs = [
            "Instruction: Classify the sentiment of this review. Data: This product is amazing, I love it!",
            "Please summarize the following article.\n\nThe study found that regular exercise can significantly reduce the risk of heart disease...",
            "Extract the names and dates from this text: John met with Sarah on January 15, 2023 to discuss the project timeline.",
            "Write a short story about a detective solving a mystery in a small town."
        ]
        
        for input_text in test_inputs:
            # Process input
            result = self.service.process_input(input_text)
            
            # Verify results
            assert "prompt" in result
            assert "data" in result
            assert "task_analysis" in result
            assert "expanded_prompt" in result
            assert "task_type" in result["task_analysis"]
            
            # Verify expansion
            assert len(result["expanded_prompt"]) > len(result["prompt"])
            assert "Role:" in result["expanded_prompt"]
            assert "Task:" in result["expanded_prompt"]
            assert "Steps:" in result["expanded_prompt"]
    
    def test_task_type_identification_accuracy(self):
        """
        Test accuracy of task type identification.
        """
        test_cases = [
            ("Classify this review as positive or negative.", "classification"),
            ("Summarize this article in three sentences.", "summarization"),
            ("Extract the main entities from this text.", "extraction"),
            ("Write a paragraph about climate change.", "generation"),
            ("What are the causes of global warming?", "question_answering"),
            ("Translate this text to French.", "translation"),
            ("Rephrase this sentence to make it more formal.", "paraphrasing"),
            ("Determine if the sentiment is positive or negative.", "sentiment_analysis"),
            ("Write a Python function to sort a list.", "code_generation")
        ]
        
        for prompt, expected_type in test_cases:
            analysis = self.analyzer.analyze(prompt)
            assert analysis["task_type"] == expected_type, f"Failed for: {prompt}"
    
    def test_prompt_data_separation_accuracy(self):
        """
        Test accuracy of prompt and data separation.
        """
        test_cases = [
            (
                "Instruction: Summarize this article. Data: The researchers found that...",
                "Summarize this article.",
                "The researchers found that..."
            ),
            (
                "Please analyze the sentiment of this review.\n\nThis product exceeded my expectations!",
                "Please analyze the sentiment of this review.",
                "This product exceeded my expectations!"
            ),
            (
                "Extract entities from: John visited Paris in June 2023.",
                "Extract entities from:",
                "John visited Paris in June 2023."
            )
        ]
        
        for input_text, expected_prompt, expected_data in test_cases:
            prompt, data = self.separator.separate(input_text)
            assert prompt.strip() == expected_prompt.strip(), f"Prompt mismatch for: {input_text}"
            assert data.strip() == expected_data.strip(), f"Data mismatch for: {input_text}"
    
    def test_prompt_expansion_quality(self):
        """
        Test quality of prompt expansion.
        """
        test_cases = [
            ("Summarize this.", "summarization", ["Role:", "Summarize", "Steps:", "Output"]),
            ("Classify this.", "classification", ["Role:", "Classify", "Steps:", "Category"]),
            ("Extract entities.", "extraction", ["Role:", "Extract", "Steps:", "information"])
        ]
        
        for prompt, task_type, expected_elements in test_cases:
            task_analysis = {
                "task_type": task_type,
                "task_confidence": 0.9,
                "category": f"{task_type}_task",
                "key_concepts": [],
                "entities": [],
                "evaluation_methods": [],
                "prompt": prompt
            }
            
            expanded = self.expander.expand(prompt, task_analysis)
            
            # Check for expected elements
            for element in expected_elements:
                assert element in expanded, f"Missing '{element}' in expansion for: {prompt}"
    
    def test_domain_adaptation_effectiveness(self):
        """
        Test effectiveness of domain adaptation.
        """
        domains = [
            ("medical article", "medical", "medical terminology"),
            ("legal document", "legal", "legal terminology"),
            ("financial report", "financial", "financial terminology"),
            ("technical documentation", "technical", "technical terminology"),
            ("scientific paper", "scientific", "scientific conventions"),
            ("educational content", "educational", "educational purposes")
        ]
        
        for concept, domain, expected_phrase in domains:
            task_analysis = {
                "task_type": "summarization",
                "task_confidence": 0.9,
                "category": "text_summarization",
                "key_concepts": [concept],
                "entities": [],
                "evaluation_methods": [],
                "prompt": f"Summarize this {concept}"
            }
            
            expanded = self.expander.expand(f"Summarize this {concept}", task_analysis)
            
            # Check for domain-specific adaptation
            assert expected_phrase in expanded.lower(), f"Missing domain adaptation for {domain}"
    
    def test_service_integration(self):
        """
        Test integration through the service layer.
        """
        # Test with different task types
        test_prompts = [
            "Classify this text as spam or not spam.",
            "Summarize this article in three sentences.",
            "Extract the key information from this text.",
            "Write a short story about space exploration."
        ]
        
        for prompt in test_prompts:
            # Test standalone expansion
            expanded = self.service.expand_prompt(prompt)
            
            # Verify expansion
            assert len(expanded) > len(prompt)
            assert "Role:" in expanded
            assert "Task:" in expanded
            assert "Steps:" in expanded
            
            # Test with full input processing
            input_text = f"Instruction: {prompt} Data: Sample content for testing."
            result = self.service.process_input(input_text)
            
            # Verify processing
            assert result["prompt"] == prompt
            assert result["data"] == "Sample content for testing."
            assert "task_type" in result["task_analysis"]
            assert len(result["expanded_prompt"]) > len(prompt)