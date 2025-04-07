"""
Evaluation metrics for LLM responses.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import re
import string
from collections import Counter
from app.utils.logger import get_logger

logger = get_logger("llm.evaluation.metrics")

class EvaluationMetrics:
    """
    Collection of metrics for evaluating LLM responses.
    """
    
    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """
        Calculate exact match score.
        
        Args:
            prediction: Predicted text.
            reference: Reference text.
            
        Returns:
            1.0 if texts match exactly (case-insensitive), 0.0 otherwise.
        """
        return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    
    @staticmethod
    def f1_score(prediction: str, reference: str) -> float:
        """
        Calculate F1 score based on token overlap.
        
        Args:
            prediction: Predicted text.
            reference: Reference text.
            
        Returns:
            F1 score between 0.0 and 1.0.
        """
        # Tokenize and normalize
        pred_tokens = EvaluationMetrics._normalize_text(prediction).split()
        ref_tokens = EvaluationMetrics._normalize_text(reference).split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0 if pred_tokens or ref_tokens else 1.0
        
        # Count common tokens
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        
        # Calculate F1
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def token_accuracy(prediction: str, reference: str) -> float:
        """
        Calculate token-level accuracy.
        
        Args:
            prediction: Predicted text.
            reference: Reference text.
            
        Returns:
            Accuracy between 0.0 and 1.0.
        """
        # Tokenize and normalize
        pred_tokens = EvaluationMetrics._normalize_text(prediction).split()
        ref_tokens = EvaluationMetrics._normalize_text(reference).split()
        
        # Handle empty cases
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Calculate accuracy
        max_len = max(len(pred_tokens), len(ref_tokens))
        correct = sum(1 for i in range(min(len(pred_tokens), len(ref_tokens))) 
                     if pred_tokens[i] == ref_tokens[i])
        
        return correct / max_len
    
    @staticmethod
    def containment_score(prediction: str, reference: str) -> float:
        """
        Calculate containment score (how much of the reference is contained in the prediction).
        
        Args:
            prediction: Predicted text.
            reference: Reference text.
            
        Returns:
            Containment score between 0.0 and 1.0.
        """
        # Tokenize and normalize
        pred_tokens = set(EvaluationMetrics._normalize_text(prediction).split())
        ref_tokens = set(EvaluationMetrics._normalize_text(reference).split())
        
        if not ref_tokens:
            return 1.0
        
        # Calculate containment
        return len(pred_tokens.intersection(ref_tokens)) / len(ref_tokens)
    
    @staticmethod
    def evaluate_classification(prediction: str, reference: str) -> Dict[str, float]:
        """
        Evaluate classification task.
        
        Args:
            prediction: Predicted text.
            reference: Reference text (expected class).
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # Extract predicted class
        pred_class = EvaluationMetrics._extract_class(prediction)
        ref_class = EvaluationMetrics._extract_class(reference)
        
        # Calculate metrics
        exact_match = 1.0 if pred_class.lower() == ref_class.lower() else 0.0
        
        return {
            "accuracy": exact_match,
            "exact_match": exact_match
        }
    
    @staticmethod
    def evaluate_extraction(prediction: str, reference: str) -> Dict[str, float]:
        """
        Evaluate extraction task.
        
        Args:
            prediction: Predicted text.
            reference: Reference text (expected extraction).
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # Calculate metrics
        f1 = EvaluationMetrics.f1_score(prediction, reference)
        exact_match = EvaluationMetrics.exact_match(prediction, reference)
        containment = EvaluationMetrics.containment_score(prediction, reference)
        
        return {
            "f1_score": f1,
            "exact_match": exact_match,
            "containment": containment,
            "overall": (f1 + containment) / 2  # Combined score
        }
    
    @staticmethod
    def evaluate_generation(prediction: str, reference: str) -> Dict[str, float]:
        """
        Evaluate text generation task.
        
        Args:
            prediction: Predicted text.
            reference: Reference text.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # Calculate metrics
        f1 = EvaluationMetrics.f1_score(prediction, reference)
        containment = EvaluationMetrics.containment_score(prediction, reference)
        
        return {
            "f1_score": f1,
            "containment": containment,
            "overall": (f1 + containment) / 2  # Combined score
        }
    
    @staticmethod
    def get_evaluator_for_task(task_type: str) -> Callable[[str, str], Dict[str, float]]:
        """
        Get an appropriate evaluator function for the specified task type.
        
        Args:
            task_type: Type of task (classification, extraction, etc.).
            
        Returns:
            Evaluation function.
        """
        evaluators = {
            "classification": EvaluationMetrics.evaluate_classification,
            "sentiment_analysis": EvaluationMetrics.evaluate_classification,
            "extraction": EvaluationMetrics.evaluate_extraction,
            "named_entity_recognition": EvaluationMetrics.evaluate_extraction,
            "summarization": EvaluationMetrics.evaluate_generation,
            "generation": EvaluationMetrics.evaluate_generation,
            "text_generation": EvaluationMetrics.evaluate_generation,
        }
        
        # Return the appropriate evaluator or a default one
        return evaluators.get(task_type, EvaluationMetrics.evaluate_generation)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for evaluation.
        
        Args:
            text: Input text.
            
        Returns:
            Normalized text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def _extract_class(text: str) -> str:
        """
        Extract class label from text.
        
        Args:
            text: Input text that may contain a class label.
            
        Returns:
            Extracted class label.
        """
        # Remove common prefixes
        prefixes = ["class:", "category:", "label:", "classification:", "result:", "the class is", "the category is"]
        for prefix in prefixes:
            if prefix in text.lower():
                text = text[text.lower().find(prefix) + len(prefix):].strip()
                break
        
        # If text is long, take first word or phrase
        if len(text.split()) > 3:
            # Check for quoted text
            match = re.search(r'"([^"]+)"', text)
            if match:
                return match.group(1)
            
            # Take first word
            return text.split()[0]
        
        return text.strip()