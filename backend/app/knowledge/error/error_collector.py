"""
Error collection module.

This module collects errors from model responses to examples.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import hashlib
import random

from app.utils.logger import get_logger
from app.core.mdp.state import PromptState

logger = get_logger("knowledge.error.error_collector")

class ErrorCollector:
    """
    Collect errors from model responses.
    
    This collector processes model responses to example inputs and
    identifies errors by comparing against expected outputs.
    """
    
    def __init__(self, llm=None):
        """
        Initialize an error collector.
        
        Args:
            llm: Optional LLM interface for testing responses.
        """
        self.llm = llm
        self.collected_errors = []
        
    async def collect_errors_async(self, prompt_state: PromptState, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect errors asynchronously using an LLM.
        
        Args:
            prompt_state: Prompt state to test.
            examples: List of examples to test.
            
        Returns:
            List of error dictionaries.
        """
        if not self.llm:
            logger.error("No LLM provided for async error collection")
            return []
            
        errors = []
        
        # Process each example
        for i, example in enumerate(examples):
            example_id = example.get("id", f"e{i+1}")
            example_text = example.get("text", "")
            expected = example.get("expected", "")
            
            if not example_text:
                continue
                
            try:
                # Combine prompt with example text
                full_prompt = f"{prompt_state.text}\n\n{example_text}"
                
                # Get model response
                response = await self.llm.generate(full_prompt)
                actual = response.get("text", "")
                
                # Check for errors
                if self._is_error(actual, expected):
                    error = self._create_error(example_id, example, actual, expected)
                    errors.append(error)
                    self.collected_errors.append(error)
            
            except Exception as e:
                logger.error(f"Error processing example {example_id}: {e}")
        
        logger.debug(f"Collected {len(errors)} errors from {len(examples)} examples")
        return errors
    
    def collect_errors(self, prompt_state: PromptState, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect errors synchronously (mock collection when no LLM is available).
        
        Args:
            prompt_state: Prompt state to test.
            examples: List of examples to test.
            
        Returns:
            List of error dictionaries.
        """
        errors = []
        
        # If LLM is available, use it
        if self.llm and hasattr(self.llm, 'generate'):
            # For testing purposes without async
            for i, example in enumerate(examples):
                example_id = example.get("id", f"e{i+1}")
                example_text = example.get("text", "")
                expected = example.get("expected", "")
                
                try:
                    # Combine prompt with example text
                    full_prompt = f"{prompt_state.text}\n\n{example_text}"
                    
                    # Try to get response if LLM supports sync
                    try:
                        response = self.llm.generate(full_prompt)
                        actual = response.get("text", "")
                    except:
                        # Mock response for testing
                        actual = self._mock_response(prompt_state, example)
                    
                    # Check for errors
                    if self._is_error(actual, expected):
                        error = self._create_error(example_id, example, actual, expected)
                        errors.append(error)
                        self.collected_errors.append(error)
                
                except Exception as e:
                    logger.error(f"Error processing example {example_id}: {e}")
        else:
            # When no LLM is available, generate mock errors for testing
            for i, example in enumerate(examples):
                example_id = example.get("id", f"e{i+1}")
                
                # Only generate errors for some examples
                if random.random() < 0.7:  # 70% chance of error for testing
                    expected = example.get("expected", "")
                    actual = self._mock_response(prompt_state, example)
                    
                    error = self._create_error(example_id, example, actual, expected)
                    errors.append(error)
                    self.collected_errors.append(error)
        
        logger.debug(f"Collected {len(errors)} errors from {len(examples)} examples")
        return errors
    
    def _is_error(self, actual: str, expected: str) -> bool:
        """
        Check if actual response differs from expected.
        
        Args:
            actual: Actual response.
            expected: Expected response.
            
        Returns:
            True if there's an error, False otherwise.
        """
        # Simple string comparison for now
        return actual.lower().strip() != expected.lower().strip()
    
    def _create_error(self, example_id: str, example: Dict[str, Any], actual: str, expected: str) -> Dict[str, Any]:
        """
        Create an error dictionary.
        
        Args:
            example_id: Example identifier.
            example: Example dictionary.
            actual: Actual response.
            expected: Expected response.
            
        Returns:
            Error dictionary.
        """
        # Generate error description
        error_type = self._determine_error_type(actual, expected)
        description = self._generate_error_description(error_type, example, actual, expected)
        
        # Create error dictionary
        error = {
            "example_id": example_id,
            "example": example,
            "error_type": error_type,
            "actual": actual,
            "expected": expected,
            "description": description
        }
        
        return error
    
    def _determine_error_type(self, actual: str, expected: str) -> str:
        """
        Determine the type of error.
        
        Args:
            actual: Actual response.
            expected: Expected response.
            
        Returns:
            Error type string.
        """
        # Simple error type determination - in real implementation this would be more sophisticated
        if not actual:
            return "empty_response"
            
        if len(actual) < len(expected) / 2:
            return "incomplete_response"
            
        if actual.lower() in expected.lower() or expected.lower() in actual.lower():
            return "partial_match"
            
        # Default to entity confusion for testing
        return "entity_confusion"
    
    def _generate_error_description(self, error_type: str, example: Dict[str, Any], actual: str, expected: str) -> str:
        """
        Generate a description of the error.
        
        Args:
            error_type: Type of error.
            example: Example dictionary.
            actual: Actual response.
            expected: Expected response.
            
        Returns:
            Error description.
        """
        example_text = example.get("text", "")
        
        if error_type == "empty_response":
            return "Model returned an empty response."
            
        if error_type == "incomplete_response":
            return "Model returned an incomplete response."
            
        if error_type == "partial_match":
            return "Model's response partially matched the expected output."
            
        if error_type == "entity_confusion":
            # For testing, generate descriptions about entity confusion
            return f"Model confused entities in the text."
        
        return f"Error in model response."
    
    def _mock_response(self, prompt_state: PromptState, example: Dict[str, Any]) -> str:
        """
        Generate a mock response for testing.
        
        Args:
            prompt_state: Prompt state.
            example: Example dictionary.
            
        Returns:
            Mock response text.
        """
        # Create a deterministic but "random" response based on the example
        example_hash = hashlib.md5(str(example).encode()).hexdigest()
        hash_int = int(example_hash, 16)
        
        expected = example.get("expected", "")
        
        # Different mock response types
        if hash_int % 3 == 0:
            # Wrong category response
            categories = ["disease", "gene", "protein", "drug", "symptom"]
            expected_lower = expected.lower()
            
            # Pick a category that's different from expected
            for cat in categories:
                if cat not in expected_lower:
                    return cat
                    
            return "incorrect_category"
        
        elif hash_int % 3 == 1:
            # Partial response
            if len(expected) > 4:
                return expected[:len(expected)//2]
            else:
                return "partial"
        
        else:
            # Completely wrong response
            return "incorrect_response"