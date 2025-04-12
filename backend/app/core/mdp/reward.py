"""
Reward function for evaluating prompt states.
"""
from typing import Dict, Any, List, Optional, Callable
from app.utils.logger import get_logger
from app.core.mdp.state import PromptState

logger = get_logger("mdp.reward")

class RewardFunction:
    """
    Reward function for evaluating prompt states.
    
    This reward function combines multiple evaluation criteria to 
    generate a single reward value for a prompt state.
    """
    
    def __init__(
        self, 
        task_performance_weight: float = 0.6,
        structural_weight: float = 0.3,
        efficiency_weight: float = 0.1,
        task_performance_fn: Optional[Callable[[PromptState], float]] = None,
        reward_booster: Optional[Callable[[PromptState, float], float]] = None
    ):
        """
        Initialize a reward function.
        
        Args:
            task_performance_weight: Weight for task performance in reward calculation.
            structural_weight: Weight for structural completeness in reward calculation.
            efficiency_weight: Weight for token efficiency in reward calculation.
            task_performance_fn: Custom function to evaluate task performance.
                If None, will use metrics["performance"] from the state.
            reward_booster: Optional function to boost reward based on task-specific criteria.
        """
        self.task_performance_weight = task_performance_weight
        self.structural_weight = structural_weight
        self.efficiency_weight = efficiency_weight
        self.task_performance_fn = task_performance_fn
        self.reward_booster = reward_booster
        
        # Ensure weights sum to 1.0
        total_weight = task_performance_weight + structural_weight + efficiency_weight
        if total_weight != 1.0:
            # Normalize weights
            self.task_performance_weight /= total_weight
            self.structural_weight /= total_weight
            self.efficiency_weight /= total_weight
        
        logger.debug(f"Initialized RewardFunction with weights: "
                     f"performance={self.task_performance_weight}, "
                     f"structure={self.structural_weight}, "
                     f"efficiency={self.efficiency_weight}")
    
    def calculate(self, state: PromptState) -> float:
        """
        Calculate the reward for a state.
        
        Args:
            state: State to evaluate.
            
        Returns:
            Reward value (higher is better).
        """
        # Evaluate performance based on metrics or custom function
        if self.task_performance_fn:
            performance = self.task_performance_fn(state)
        else:
            performance = state.metrics.get("performance", 0.0)
        
        # Evaluate structural completeness
        structural_completeness = state.get_structural_completeness()
        
        # Evaluate token efficiency
        token_efficiency = state.get_token_efficiency()
        
        # Calculate weighted sum
        reward = (
            self.task_performance_weight * performance +
            self.structural_weight * structural_completeness +
            self.efficiency_weight * token_efficiency
        )
        
        # Apply reward booster if available
        if self.reward_booster:
            boost = self.reward_booster(state, reward)
            reward += boost
        
        logger.debug(f"Calculated reward {reward:.4f} for state {state}: "
                     f"performance={performance:.4f}, "
                     f"structure={structural_completeness:.4f}, "
                     f"efficiency={token_efficiency:.4f}")
        
        return reward
    
    def get_component_rewards(self, state: PromptState) -> Dict[str, float]:
        """
        Get individual reward components for a state.
        
        Args:
            state: State to evaluate.
            
        Returns:
            Dictionary with component rewards.
        """
        # Evaluate performance based on metrics or custom function
        if self.task_performance_fn:
            performance = self.task_performance_fn(state)
        else:
            performance = state.metrics.get("performance", 0.0)
        
        # Evaluate structural completeness
        structural_completeness = state.get_structural_completeness()
        
        # Evaluate token efficiency
        token_efficiency = state.get_token_efficiency()
        
        return {
            "performance": performance,
            "structural_completeness": structural_completeness,
            "token_efficiency": token_efficiency,
            "weighted_performance": self.task_performance_weight * performance,
            "weighted_structural": self.structural_weight * structural_completeness,
            "weighted_efficiency": self.efficiency_weight * token_efficiency,
            "total": (self.task_performance_weight * performance +
                     self.structural_weight * structural_completeness +
                     self.efficiency_weight * token_efficiency)
        }
    
    def with_task_performance_fn(self, fn: Callable[[PromptState], float]) -> 'RewardFunction':
        """
        Create a new RewardFunction with a custom task performance function.
        
        Args:
            fn: Function that takes a state and returns a performance score.
            
        Returns:
            New RewardFunction instance.
        """
        return RewardFunction(
            task_performance_weight=self.task_performance_weight,
            structural_weight=self.structural_weight,
            efficiency_weight=self.efficiency_weight,
            task_performance_fn=fn
        )


class PerformanceEvaluator:
    """
    Evaluates the performance of prompts for specific task types.
    
    This class provides task-specific performance evaluation functions.
    """
    
    @staticmethod
    def classification_evaluator(state: PromptState) -> float:
        """
        Evaluate performance for classification tasks.
        
        Args:
            state: State to evaluate.
            
        Returns:
            Performance score between 0.0 and 1.0.
        """
        # In a real implementation, this would use external evaluation results.
        # For now, we use a heuristic based on structure.
        components = state.components
        
        score = 0.0
        
        # Check for classification-specific elements - increase the weights
        if "role" in components and any(term in str(components.get("role", "")).lower() 
                                     for term in ["classification", "sentiment"]):
            score += 0.15
            
        if "task" in components and any(term in str(components.get("task", "")).lower() 
                                      for term in ["classify", "sentiment", "analyze"]):
            score += 0.15
            
        if "output_format" in components and any(term in str(components.get("output_format", "")).lower() 
                                               for term in ["category", "positive", "negative", "neutral", "sentiment"]):
            score += 0.25
            
        if "steps" in components:
            steps = " ".join([str(s) for s in components["steps"]]).lower()
            classification_terms = ["analyze", "assign", "category", "sentiment", "positive", "negative"]
            step_score = sum(0.05 for term in classification_terms if term in steps)
            score += min(0.3, step_score)
                
        # Basic structural check
        score += min(0.3, state.get_structural_completeness())
        
        return min(1.0, score)
    
    @staticmethod
    def extraction_evaluator(state: PromptState) -> float:
        """
        Evaluate performance for extraction tasks.
        
        Args:
            state: State to evaluate.
            
        Returns:
            Performance score between 0.0 and 1.0.
        """
        # In a real implementation, this would use external evaluation results.
        # For now, we use a heuristic based on structure.
        components = state.components
        
        score = 0.0
        
        # Check for extraction-specific elements
        if "role" in components and any(term in components["role"].lower() 
                                       for term in ["extract", "finder", "identification"]):
            score += 0.1
            
        if "task" in components and any(term in components["task"].lower() 
                                      for term in ["extract", "find", "identify"]):
            score += 0.1
            
        if "output_format" in components and any(term in components["output_format"].lower() 
                                               for term in ["list", "key-value", "structured"]):
            score += 0.2
            
        if "steps" in components:
            steps = " ".join(components["steps"]).lower()
            extraction_terms = ["scan", "locate", "identify", "extract", "present"]
            step_score = sum(0.1 for term in extraction_terms if term in steps)
            score += min(0.3, step_score)
                
        # Basic structural check
        score += min(0.3, state.get_structural_completeness())
        
        return min(1.0, score)
    
    @staticmethod
    def summarization_evaluator(state: PromptState) -> float:
        """
        Evaluate performance for summarization tasks.
        
        Args:
            state: State to evaluate.
            
        Returns:
            Performance score between 0.0 and 1.0.
        """
        # In a real implementation, this would use external evaluation results.
        # For now, we use a heuristic based on structure.
        components = state.components
        
        score = 0.0
        
        # Check for summarization-specific elements
        if "role" in components and any(term in components["role"].lower() 
                                       for term in ["summar", "condense", "digest"]):
            score += 0.1
            
        if "task" in components and any(term in components["task"].lower() 
                                      for term in ["summar", "condense", "main points"]):
            score += 0.1
            
        if "output_format" in components and any(term in components["output_format"].lower() 
                                               for term in ["concise", "brief", "paragraph", "bullet"]):
            score += 0.2
            
        if "steps" in components:
            steps = " ".join(components["steps"]).lower()
            summary_terms = ["read", "identify", "key", "main", "point", "concise", "complete"]
            step_score = sum(0.05 for term in summary_terms if term in steps)
            score += min(0.3, step_score)
                
        # Basic structural check
        score += min(0.3, state.get_structural_completeness())
        
        return min(1.0, score)
    
    @staticmethod
    def generation_evaluator(state: PromptState) -> float:
        """
        Evaluate performance for generation tasks.
        
        Args:
            state: State to evaluate.
            
        Returns:
            Performance score between 0.0 and 1.0.
        """
        # In a real implementation, this would use external evaluation results.
        # For now, we use a heuristic based on structure.
        components = state.components
        
        score = 0.0
        
        # Check for generation-specific elements
        if "role" in components and any(term in components["role"].lower() 
                                       for term in ["creator", "writer", "author", "composer"]):
            score += 0.1
            
        if "task" in components and any(term in components["task"].lower() 
                                      for term in ["generate", "create", "write", "compose"]):
            score += 0.1
            
        if "output_format" in components and any(term in components["output_format"].lower() 
                                               for term in ["structure", "format", "section", "paragraph"]):
            score += 0.2
            
        if "steps" in components:
            steps = " ".join(components["steps"]).lower()
            generation_terms = ["plan", "outline", "draft", "develop", "structure", "review", "edit"]
            step_score = sum(0.05 for term in generation_terms if term in steps)
            score += min(0.3, step_score)
                
        # Basic structural check
        score += min(0.3, state.get_structural_completeness())
        
        return min(1.0, score)
    
    @staticmethod
    def get_evaluator_for_task(task_type: str) -> Callable[[PromptState], float]:
        """
        Get an appropriate evaluator function for the specified task type.
        
        Args:
            task_type: Type of task (classification, extraction, etc.).
            
        Returns:
            Evaluation function.
        """
        evaluators = {
            "classification": PerformanceEvaluator.classification_evaluator,
            "sentiment_analysis": PerformanceEvaluator.classification_evaluator,  # Use classification evaluator
            "extraction": PerformanceEvaluator.extraction_evaluator,
            "named_entity_recognition": PerformanceEvaluator.extraction_evaluator,  # Use extraction evaluator
            "summarization": PerformanceEvaluator.summarization_evaluator,
            "generation": PerformanceEvaluator.generation_evaluator,
            "text_generation": PerformanceEvaluator.generation_evaluator,  # Use generation evaluator
            "story_generation": PerformanceEvaluator.generation_evaluator,  # Use generation evaluator
        }
        
        # Return the appropriate evaluator or a default one
        return evaluators.get(task_type, PerformanceEvaluator.default_evaluator)
    
    @staticmethod
    def default_evaluator(state: PromptState) -> float:
        """
        Default performance evaluator for any task.
        
        Args:
            state: State to evaluate.
            
        Returns:
            Performance score between 0.0 and 1.0.
        """
        # In a real implementation, this would use external evaluation results.
        # For now, we use a basic structural evaluation.
        
        # Get structural completeness
        structural_score = state.get_structural_completeness()
        
        # Check for basic prompt elements
        components = state.components
        element_score = 0.0
        
        if "role" in components and components["role"]:
            element_score += 0.2
            
        if "task" in components and components["task"]:
            element_score += 0.3
            
        if "steps" in components and components["steps"]:
            element_score += 0.3
            
        if "output_format" in components and components["output_format"]:
            element_score += 0.2
        
        # Combine scores
        return 0.6 * structural_score + 0.4 * element_score