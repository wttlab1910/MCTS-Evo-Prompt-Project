"""
Final output generation service for optimized prompts.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from app.core.mdp.state import PromptState
from app.core.mcts.node import MCTSNode
from app.core.optimization.prompt_selector import PromptSelector
from app.core.optimization.token_optimizer import TokenOptimizer
from app.core.optimization.output_processor import OutputProcessor
from app.utils.logger import get_logger

logger = get_logger("services.output_service")

class OutputService:
    """
    Service for handling final output generation and processing.
    
    This service integrates the various optimization components to produce
    the final optimized prompt.
    """
    
    def __init__(self):
        """Initialize the output service."""
        self.prompt_selector = PromptSelector()
        self.token_optimizer = TokenOptimizer()
        self.output_processor = OutputProcessor(token_optimizer=self.token_optimizer)
        
        logger.info("Output service initialized.")
    
    def generate_output(self, 
                      root_node: MCTSNode, 
                      original_data: Optional[str] = None,
                      selection_strategy: str = "composite",
                      verification_level: str = "standard") -> Dict[str, Any]:
        """
        Generate the final optimized output from an MCTS search tree.
        
        Args:
            root_node: Root node of the MCTS search tree.
            original_data: Original data to combine with the prompt (optional).
            selection_strategy: Strategy for selecting the optimal prompt.
            verification_level: Level of verification for quality assurance.
            
        Returns:
            Dictionary with the optimized output and detailed statistics.
        """
        # 修复: 移除logger.block调用
        logger.info("Generating final output")
        
        # Select the optimal prompt
        best_state, selection_stats = self.prompt_selector.select_optimal_prompt(
            root_node, strategy=selection_strategy
        )
        
        # Analyze top trajectories
        top_trajectories = self.prompt_selector.analyze_trajectories(root_node, top_k=3)
        
        # Process the output
        final_output, processing_stats = self.output_processor.process_output(
            optimized_state=best_state,
            original_data=original_data,
            verification_level=verification_level
        )
        
        # Compile the complete result
        result = {
            "final_output": final_output,
            "best_state": best_state.text,
            "selection_stats": selection_stats,
            "processing_stats": processing_stats,
            "top_trajectories": [
                {
                    "path_score": t["evaluation"]["path_score"],
                    "leaf_reward": t["leaf_node"].avg_reward,
                    "path_length": t["evaluation"]["path_length"]
                } for t in top_trajectories
            ],
            "success": processing_stats["verification_passed"]
        }
        
        logger.info(f"Output generation complete: {len(final_output)} characters")
        
        return result

    
    def compare_with_original(self, 
                            original_prompt: str, 
                            optimized_prompt: str) -> Dict[str, Any]:
        """
        Compare the optimized prompt with the original prompt.
        
        Args:
            original_prompt: The original prompt text.
            optimized_prompt: The optimized prompt text.
            
        Returns:
            Dictionary with comparison metrics.
        """
        # Create prompt states for analysis
        original_state = PromptState(original_prompt)
        optimized_state = PromptState(optimized_prompt)
        
        # Calculate basic metrics
        original_length = len(original_prompt)
        optimized_length = len(optimized_prompt)
        length_diff = optimized_length - original_length
        length_diff_percent = (length_diff / original_length) * 100 if original_length > 0 else 0
        
        # Compare component presence
        component_types = {"role", "task", "steps", "output_format", "examples", "constraints"}
        original_components = {comp: original_state.has_component(comp) for comp in component_types}
        optimized_components = {comp: optimized_state.has_component(comp) for comp in component_types}
        
        # Calculate component differences
        component_added = sum(1 for comp in component_types if not original_components[comp] and optimized_components[comp])
        component_removed = sum(1 for comp in component_types if original_components[comp] and not optimized_components[comp])
        
        # Extract structural improvements
        structural_improvements = []
        
        if not original_components["role"] and optimized_components["role"]:
            structural_improvements.append("Added expert role")
        
        if not original_components["steps"] and optimized_components["steps"]:
            structural_improvements.append("Added step-by-step instructions")
        
        if not original_components["output_format"] and optimized_components["output_format"]:
            structural_improvements.append("Added output format specification")
        
        if not original_components["examples"] and optimized_components["examples"]:
            structural_improvements.append("Added examples")
        
        # Compare original vs optimized steps (if present)
        step_comparison = None
        if optimized_components["steps"]:
            original_step_count = len(original_state.components.get("steps", [])) if isinstance(original_state.components.get("steps", []), list) else 0
            optimized_step_count = len(optimized_state.components.get("steps", [])) if isinstance(optimized_state.components.get("steps", []), list) else 0
            
            step_comparison = {
                "original_count": original_step_count,
                "optimized_count": optimized_step_count,
                "difference": optimized_step_count - original_step_count
            }
            
            if optimized_step_count > original_step_count:
                structural_improvements.append(f"Increased step count from {original_step_count} to {optimized_step_count}")
        
        return {
            "original_length": original_length,
            "optimized_length": optimized_length,
            "length_difference": length_diff,
            "length_difference_percent": length_diff_percent,
            "original_components": original_components,
            "optimized_components": optimized_components,
            "components_added": component_added,
            "components_removed": component_removed,
            "structural_improvements": structural_improvements,
            "step_comparison": step_comparison
        }