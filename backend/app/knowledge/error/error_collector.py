"""
Error collection mechanism for prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import random
from app.core.mdp.state import PromptState
from app.utils.logger import get_logger

logger = get_logger("error.collector")

class ErrorCollector:
    """
    Collects errors from LLM responses using the current prompt.
    
    This class handles sampling examples, processing them with the base LLM,
    and identifying cases where the model's output differs from the expected outcome.
    """
    
    def __init__(
        self,
        llm_interface=None,  # This would be a reference to your LLM interface
        sample_size: int = 5,
        stratified_sampling: bool = True
    ):
        """
        Initialize an error collector.
        
        Args:
            llm_interface: Interface to the LLM for processing examples.
            sample_size: Number of examples to sample for error collection.
            stratified_sampling: Whether to use stratified sampling for balanced examples.
        """
        self.llm_interface = llm_interface
        self.sample_size = sample_size
        self.stratified_sampling = stratified_sampling
        logger.debug(f"Initialized ErrorCollector with sample_size={sample_size}, "
                    f"stratified_sampling={stratified_sampling}")
    
    def collect_errors(
        self,
        prompt_state: PromptState,
        examples: List[Dict[str, Any]],
        expected_outputs: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect errors by processing examples with the current prompt.
        
        Args:
            prompt_state: Current prompt state to evaluate.
            examples: List of example inputs.
            expected_outputs: Optional list of expected outputs for each example.
            
        Returns:
            List of error information dictionaries.
        """
        # If no LLM interface is provided, use mock errors for development/testing
        if self.llm_interface is None:
            return self._generate_mock_errors(prompt_state, examples)
        
        # Sample examples for error collection
        sampled_examples = self._sample_examples(examples)
        
        errors = []
        for i, example in enumerate(sampled_examples):
            # Get expected output if available
            expected = expected_outputs[i] if expected_outputs and i < len(expected_outputs) else None
            
            # Process the example with the current prompt
            try:
                # This would call the actual LLM in a real implementation
                actual_output = self.llm_interface.process(prompt_state.text, example)
                
                # Compare with expected output if available
                if expected is not None and not self._outputs_match(actual_output, expected):
                    # Record error
                    errors.append({
                        "example_id": i,
                        "example": example,
                        "expected": expected,
                        "actual": actual_output,
                        "error_type": "output_mismatch"
                    })
            except Exception as e:
                # Record processing error
                errors.append({
                    "example_id": i,
                    "example": example,
                    "expected": expected,
                    "error_type": "processing_error",
                    "error_message": str(e)
                })
        
        logger.debug(f"Collected {len(errors)} errors from {len(sampled_examples)} examples")
        return errors
    
    def _sample_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sample examples for error collection.
        
        Args:
            examples: List of all available examples.
            
        Returns:
            List of sampled examples.
        """
        if len(examples) <= self.sample_size:
            return examples
        
        if self.stratified_sampling:
            # In a real implementation, this would group examples by relevant characteristics
            # For now, just do random sampling
            return random.sample(examples, self.sample_size)
        else:
            return random.sample(examples, self.sample_size)
    
    def _outputs_match(self, actual: Any, expected: Any) -> bool:
        """
        Check if actual and expected outputs match.
        
        Args:
            actual: Actual output from the LLM.
            expected: Expected output.
            
        Returns:
            True if outputs match, False otherwise.
        """
        # In a real implementation, this would use more sophisticated matching
        # based on the specific task (e.g., semantic similarity for summarization)
        return str(actual).strip() == str(expected).strip()
    
    def _generate_mock_errors(
        self,
        prompt_state: PromptState,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate mock errors for development and testing purposes.
        
        Args:
            prompt_state: Current prompt state.
            examples: List of example inputs.
            
        Returns:
            List of mock error information dictionaries.
        """
        # Analysis of the prompt to determine likely error types
        has_clear_format = "output_format" in prompt_state.components
        has_examples = "examples" in prompt_state.components
        has_steps = "steps" in prompt_state.components and prompt_state.components["steps"]
        
        # Generate mock errors based on prompt characteristics
        mock_errors = []
        
        # Sample a subset of examples
        sampled_indices = random.sample(range(min(5, len(examples))), min(3, len(examples)))
        
        for i in sampled_indices:
            example = examples[i]
            
            # Types of potential errors
            error_types = []
            
            if not has_clear_format:
                error_types.append("format_error")
            
            if not has_examples:
                error_types.append("content_error")
            
            if not has_steps:
                error_types.append("reasoning_error")
            
            # Always include general errors
            error_types.extend(["omission_error", "hallucination_error"])
            
            # Select an error type
            error_type = random.choice(error_types)
            
            # Create mock error
            mock_error = {
                "example_id": i,
                "example": example,
                "expected": "Expected output for example " + str(i),
                "actual": "Incorrect output with " + error_type,
                "error_type": error_type
            }
            
            mock_errors.append(mock_error)
        
        logger.debug(f"Generated {len(mock_errors)} mock errors")
        return mock_errors
    
    def set_sample_size(self, size: int) -> None:
        """
        Set the sample size for error collection.
        
        Args:
            size: New sample size.
        """
        self.sample_size = max(1, size)
        logger.debug(f"Set sample size to {self.sample_size}")
    
    def set_stratified_sampling(self, stratified: bool) -> None:
        """
        Set whether to use stratified sampling.
        
        Args:
            stratified: True to use stratified sampling, False otherwise.
        """
        self.stratified_sampling = stratified
        logger.debug(f"Set stratified sampling to {self.stratified_sampling}")