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
            
            # Handle examples with different formats
            if isinstance(example, dict) and "question" in example and "answer" in example:
                example_text = example.get("question", "")
                expected = example.get("answer", "")
            else:
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
                # Handle examples with different formats
                if isinstance(example, dict) and "question" in example and "answer" in example:
                    example_id = example.get("id", f"e{i+1}")
                    example_text = example.get("question", "")
                    expected = example.get("answer", "")
                else:
                    example_id = example.get("id", f"e{i+1}")
                    example_text = example.get("text", "")
                    expected = example.get("expected", "")
                
                if not example_text:
                    continue
                
                try:
                    # Combine prompt with example text
                    full_prompt = f"{prompt_state.text}\n\n{example_text}"
                    
                    # Try to get response if LLM supports sync
                    try:
                        # Use loop.run_until_complete for async calls
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                import nest_asyncio
                                nest_asyncio.apply()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                        # Call the async method
                        response = loop.run_until_complete(self.llm.generate(full_prompt))
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
                if isinstance(example, dict):
                    example_id = example.get("id", f"e{i+1}")
                    
                    # Only generate errors for some examples
                    if random.random() < 0.7:  # 70% chance of error for testing
                        if "answer" in example:
                            expected = example.get("answer", "")
                        else:
                            expected = example.get("expected", "")
                            
                        actual = self._mock_response(prompt_state, example)
                        
                        error = self._create_error(example_id, example, actual, expected)
                        errors.append(error)
                        self.collected_errors.append(error)
        
        logger.debug(f"Collected {len(errors)} errors from {len(examples)} examples")
        return errors
    
    def process_errors_from_evaluation(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process errors from evaluation results.
        
        Args:
            errors: List of error dictionaries from evaluation.
            
        Returns:
            Processed error dictionaries.
        """
        processed_errors = []
        
        for error in errors:
            example_id = error.get("id", f"e{len(processed_errors)}")
            example_text = error.get("text", "")
            expected = error.get("expected", "")
            actual = error.get("actual", "")
            error_type = error.get("error_type", self._determine_error_type(actual, expected))
            
            processed_error = self._create_error(example_id, {"text": example_text}, actual, expected, error_type)
            processed_errors.append(processed_error)
            self.collected_errors.append(processed_error)
        
        logger.debug(f"Processed {len(processed_errors)} errors from evaluation")
        return processed_errors
    
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
    
    def _create_error(self, example_id: str, example: Dict[str, Any], actual: str, expected: str, error_type: str = None) -> Dict[str, Any]:
        """
        Create an error dictionary.
        
        Args:
            example_id: Example identifier.
            example: Example dictionary.
            actual: Actual response.
            expected: Expected response.
            error_type: Type of error (optional).
            
        Returns:
            Error dictionary.
        """
        # Generate error description
        if not error_type:
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
        # More sophisticated error type determination
        if not actual:
            return "empty_response"
            
        if len(actual) < len(expected) / 2:
            return "incomplete_response"
            
        if actual.lower() in expected.lower() or expected.lower() in actual.lower():
            return "partial_match"
            
        # Check for table-related errors
        if any(word in expected.lower() for word in ["table", "column", "row", "count", "average"]):
            if "count" in expected.lower() or any(c.isdigit() for c in expected):
                return "calculation_error"
            else:
                return "lookup_error"
                
        # Check for numerical errors
        if any(c.isdigit() for c in expected) and any(c.isdigit() for c in actual):
            return "numerical_error"
            
        # Check for classification errors
        if expected.lower() in ["yes", "no", "true", "false"]:
            return "classification_error"
            
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
        if not example_text and "question" in example:
            example_text = example.get("question", "")
        
        if error_type == "empty_response":
            return "Model returned an empty response when an answer was expected."
            
        if error_type == "incomplete_response":
            return "Model returned an incomplete response, missing key information."
            
        if error_type == "partial_match":
            return "Model's response partially matched the expected output but was not fully correct."
            
        if error_type == "lookup_error":
            return "Model failed to correctly look up the required information from the table."
            
        if error_type == "calculation_error":
            return "Model made an error in calculation or counting when processing numerical data."
            
        if error_type == "numerical_error":
            return "Model provided an incorrect numerical answer."
            
        if error_type == "classification_error":
            return "Model incorrectly classified or categorized the information."
            
        if error_type == "entity_confusion":
            return "Model confused entities or concepts in the text."
        
        return f"Error in model response: expected '{expected}' but got '{actual}'."
    
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
        
        # Get expected answer from different possible formats
        if "answer" in example:
            expected = example.get("answer", "")
        else:
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