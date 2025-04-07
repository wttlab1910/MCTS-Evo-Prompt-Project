"""
Validator for LLM responses.
"""
from typing import Dict, Any, List, Optional, Union, Callable
from app.utils.logger import get_logger
from app.llm.evaluation.metrics import EvaluationMetrics

logger = get_logger("llm.evaluation.validator")

class ResponseValidator:
    """
    Validator for LLM responses.
    
    This class provides methods to validate LLM responses and calculate quality metrics.
    """
    
    def __init__(self, task_type: Optional[str] = None):
        """
        Initialize the response validator.
        
        Args:
            task_type: Type of task for specialized validation.
        """
        self.task_type = task_type
        self.evaluator = EvaluationMetrics.get_evaluator_for_task(task_type) if task_type else None
        
        logger.debug(f"Initialized ResponseValidator for task_type={task_type}")
    
    def validate(self, 
                response: Dict[str, Any], 
                expected: Optional[str] = None,
                criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate a response against criteria and expected output.
        
        Args:
            response: Response dictionary from LLM.
            expected: Expected output (optional).
            criteria: Validation criteria (optional).
            
        Returns:
            Dictionary with validation results.
        """
        # Initialize validation result
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Check for errors in response
        if "error" in response and response["error"]:
            validation["valid"] = False
            validation["errors"].append(f"LLM error: {response['error']}")
        
        # Check for empty or very short responses
        if not response.get("text") or len(response.get("text", "")) < 5:
            validation["valid"] = False
            validation["errors"].append("Response is empty or too short")
        
        # Check finish reason
        if response.get("finish_reason") != "stop":
            if response.get("finish_reason") == "length":
                validation["warnings"].append("Response was truncated due to length limits")
            else:
                validation["warnings"].append(f"Unusual finish reason: {response.get('finish_reason')}")
        
        # Apply custom criteria if provided
        if criteria:
            self._apply_criteria(response, criteria, validation)
        
        # Calculate metrics if expected output is provided
        if expected and "text" in response:
            if self.evaluator:
                # Use task-specific evaluator
                validation["metrics"] = self.evaluator(response["text"], expected)
            else:
                # Use general metrics
                validation["metrics"] = {
                    "exact_match": EvaluationMetrics.exact_match(response["text"], expected),
                    "f1_score": EvaluationMetrics.f1_score(response["text"], expected),
                    "containment": EvaluationMetrics.containment_score(response["text"], expected)
                }
            
            # Add a simple quality score
            if "overall" in validation["metrics"]:
                validation["quality_score"] = validation["metrics"]["overall"]
            else:
                # Average available metrics
                metrics = validation["metrics"]
                validation["quality_score"] = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        return validation
    
    def _apply_criteria(self, 
                       response: Dict[str, Any], 
                       criteria: Dict[str, Any],
                       validation: Dict[str, Any]) -> None:
        """
        Apply custom validation criteria.
        
        Args:
            response: Response dictionary from LLM.
            criteria: Validation criteria.
            validation: Validation result to update.
        """
        response_text = response.get("text", "")
        
        # Check for required content
        if "required_content" in criteria:
            for content in criteria["required_content"]:
                if content.lower() not in response_text.lower():
                    validation["valid"] = False
                    validation["errors"].append(f"Missing required content: {content}")
        
        # Check for forbidden content
        if "forbidden_content" in criteria:
            for content in criteria["forbidden_content"]:
                if content.lower() in response_text.lower():
                    validation["valid"] = False
                    validation["errors"].append(f"Contains forbidden content: {content}")
        
        # Check for minimum length
        if "min_length" in criteria and len(response_text) < criteria["min_length"]:
            validation["valid"] = False
            validation["errors"].append(f"Response too short: {len(response_text)} < {criteria['min_length']}")
        
        # Check for maximum length
        if "max_length" in criteria and len(response_text) > criteria["max_length"]:
            validation["warnings"].append(f"Response too long: {len(response_text)} > {criteria['max_length']}")
        
        # Check for required format
        if "format_regex" in criteria:
            import re
            if not re.search(criteria["format_regex"], response_text):
                validation["valid"] = False
                validation["errors"].append("Response does not match required format")