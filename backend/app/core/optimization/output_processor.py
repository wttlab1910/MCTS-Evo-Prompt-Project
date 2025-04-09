"""
Final output processing for prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from app.core.mdp.state import PromptState
from app.core.optimization.token_optimizer import TokenOptimizer
from app.utils.logger import get_logger

logger = get_logger("optimization.output_processor")

class OutputProcessor:
    """
    Process optimized prompts for final output and deployment.
    
    This class handles reconstruction, verification, formatting, and quality assurance
    of prompts after optimization.
    """
    
    def __init__(self, token_optimizer: Optional[TokenOptimizer] = None):
        """
        Initialize an output processor.
        
        Args:
            token_optimizer: Optional token optimizer for efficiency optimization.
        """
        self.token_optimizer = token_optimizer or TokenOptimizer()
        logger.debug("Initialized OutputProcessor")
    
    def process_output(self, 
                      optimized_state: PromptState, 
                      original_data: Optional[str] = None,
                      verification_level: str = "standard") -> Tuple[str, Dict[str, Any]]:
        """
        Process an optimized prompt for final output.
        
        Args:
            optimized_state: The optimized prompt state.
            original_data: Original data to combine with the prompt (optional).
            verification_level: Level of verification ("minimal", "standard", "thorough").
            
        Returns:
            Tuple of (final_output, processing_stats).
        """
        logger.info(f"Processing output with verification level: {verification_level}")
        
        # Initialize stats
        stats = {
            "verification_level": verification_level,
            "verification_passed": True,
            "verification_issues": [],
            "token_optimization_applied": False,
            "token_optimization_stats": {},
            "components_present": {},
            "final_length": 0,
        }
        
        # Apply token optimization if needed
        if self.token_optimizer:
            optimized_state, token_stats = self.token_optimizer.optimize(optimized_state)
            stats["token_optimization_applied"] = True
            stats["token_optimization_stats"] = token_stats
        
        # Verify the prompt's quality
        verification_result = self._verify_prompt(optimized_state, level=verification_level)
        stats["verification_passed"] = verification_result["passed"]
        stats["verification_issues"] = verification_result["issues"]
        stats["components_present"] = verification_result["components"]
        
        # Reconstruct the final output
        final_output = self._reconstruct_prompt(optimized_state, original_data)
        
        # Format for deployment
        final_output = self._format_for_deployment(final_output, optimized_state)
        
        # Final quality check
        final_output = self._apply_quality_assurance(final_output, verification_result)
        
        stats["final_length"] = len(final_output)
        logger.info(f"Output processing complete, final length: {len(final_output)}")
        
        return final_output, stats
    
    def _verify_prompt(self, state: PromptState, level: str = "standard") -> Dict[str, Any]:
        """
        Verify the quality and completeness of a prompt.
        
        Args:
            state: Prompt state to verify.
            level: Verification level.
            
        Returns:
            Verification result dictionary.
        """
        verification = {
            "passed": True,
            "issues": [],
            "components": {}
        }
        
        # Check for essential components
        essential_components = ["role", "task"]
        if level == "thorough":
            essential_components.extend(["steps", "output_format"])
        elif level == "standard":
            essential_components.append("steps")
        
        # Check each component
        for component in essential_components:
            component_present = state.has_component(component)
            verification["components"][component] = component_present
            
            if not component_present:
                verification["passed"] = False
                verification["issues"].append(f"Missing essential component: {component}")
        
        # Additional checks based on verification level
        if level == "thorough" or level == "standard":
            # Check for empty or very short components
            for component, text in state.components.items():
                if isinstance(text, str) and len(text.strip()) < 5:
                    verification["issues"].append(f"Component '{component}' is too short")
                elif isinstance(text, list) and (not text or all(len(item.strip()) < 5 for item in text)):
                    verification["issues"].append(f"Component '{component}' contains empty or very short items")
            
            # Check for overly long components
            text = state.text
            if len(text) > 2000:
                verification["issues"].append("Prompt is excessively long (>2000 characters)")
            
            # Check for redundancy
            for component1, text1 in state.components.items():
                if isinstance(text1, str):
                    for component2, text2 in state.components.items():
                        if component1 != component2 and isinstance(text2, str):
                            # Check if one component is substantially contained within another
                            if len(text1) > 20 and len(text2) > 20 and text1 in text2:
                                verification["issues"].append(f"Redundancy detected: {component1} is contained within {component2}")
        
        if level == "thorough":
            # Check formatting consistency
            if "steps" in state.components and isinstance(state.components["steps"], list):
                steps = state.components["steps"]
                
                # Check for consistent step prefixes
                step_formats = []
                for step in steps:
                    if re.match(r'^\d+\.', step):
                        step_formats.append("numbered")
                    elif step.startswith('-'):
                        step_formats.append("bullet")
                    else:
                        step_formats.append("plain")
                
                if len(set(step_formats)) > 1:
                    verification["issues"].append("Inconsistent step formatting (mix of numbered, bullet, plain)")
            
            # Check language style consistency
            imperative_count = len(re.findall(r'\b(analyze|identify|determine|calculate|evaluate)\b', state.text, re.IGNORECASE))
            passive_count = len(re.findall(r'\b(should be|must be|is to be|are to be)\b', state.text, re.IGNORECASE))
            
            if imperative_count > 0 and passive_count > 0:
                verification["issues"].append("Mixed instruction styles (both imperative and passive)")
        
        return verification
    
    def _reconstruct_prompt(self, state: PromptState, original_data: Optional[str] = None) -> str:
        """
        Reconstruct the final prompt, optionally with original data.
        
        Args:
            state: Prompt state to reconstruct.
            original_data: Original data to include (optional).
            
        Returns:
            Reconstructed prompt string.
        """
        # Get the optimized prompt text
        prompt = state.text.strip()
        
        # If no data, just return the prompt
        if not original_data:
            return prompt
        
        # Determine appropriate separator based on prompt structure
        separator = "\n\n"
        
        # For structured prompts with specific components, use more explicit separation
        if state.has_component("output_format") and "Content:" not in prompt:
            # Add an explicit Content label if not already present
            prompt += "\n\nContent:"
            separator = " "
        
        # Combine prompt with data
        return f"{prompt}{separator}{original_data}"
    
    def _format_for_deployment(self, output: str, state: PromptState) -> str:
        """
        Format the output for deployment.
        
        Args:
            output: Output string to format.
            state: Original prompt state.
            
        Returns:
            Formatted output string.
        """
        # Ensure consistent line breaks
        formatted = output.replace('\r\n', '\n')
        
        # Add a trailing new line if not present
        if not formatted.endswith('\n'):
            formatted += '\n'
        
        # Ensure double spacing between sections
        formatted = re.sub(r'([.!?:])\n([A-Z])', r'\1\n\n\2', formatted)
        
        # Fix any triple or more line breaks
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
        # Add comment separators for major sections if not already present
        if state.has_component("role") and not re.search(r'-{3,}', formatted):
            formatted = re.sub(r'(Role:.*?)(\n\n|\n[A-Z])', r'\1\n\n---\n\2', formatted, count=1, flags=re.DOTALL)
        
        return formatted
    
    def _apply_quality_assurance(self, output: str, verification: Dict[str, Any]) -> str:
        """
        Apply quality assurance fixes based on verification issues.
        
        Args:
            output: Output string to fix.
            verification: Verification result dictionary.
            
        Returns:
            Fixed output string.
        """
        fixed_output = output
        
        # Fix specific issues identified during verification
        for issue in verification["issues"]:
            if "Missing essential component" in issue:
                component = issue.split(": ")[1]
                if component == "role" and "Role:" not in fixed_output:
                    fixed_output = f"Role: Expert\n\n{fixed_output}"
                
                elif component == "task" and "Task:" not in fixed_output:
                    if "Role:" in fixed_output:
                        fixed_output = re.sub(r'(Role:.*?)(\n\n|\n[A-Z])', r'\1\n\nTask: Complete the assigned task effectively\2', fixed_output, count=1, flags=re.DOTALL)
                    else:
                        fixed_output = f"Task: Complete the assigned task effectively\n\n{fixed_output}"
                
                elif component == "steps" and "Steps:" not in fixed_output and "Process:" not in fixed_output:
                    if "Task:" in fixed_output:
                        fixed_output = re.sub(r'(Task:.*?)(\n\n|\n[A-Z])', 
                                           r'\1\n\nSteps:\n- Analyze the input\n- Process the information\n- Generate the output\2', 
                                           fixed_output, count=1, flags=re.DOTALL)
                
                elif component == "output_format" and "Output Format:" not in fixed_output and "Output:" not in fixed_output:
                    fixed_output += "\n\nOutput Format: Provide a clear and structured response."
            
            elif "Inconsistent step formatting" in issue:
                # Standardize step formatting to bullets
                step_section = re.search(r'Steps:(.*?)(\n\n|\n[A-Z]|$)', fixed_output, re.DOTALL)
                if step_section:
                    steps_text = step_section.group(1)
                    lines = steps_text.strip().split('\n')
                    
                    # Convert to bullet format
                    new_steps = ["Steps:"]
                    for line in lines:
                        # Remove any existing numbering or bullets
                        cleaned_line = re.sub(r'^\s*[\d-]+\.?\s*', '', line).strip()
                        if cleaned_line:
                            new_steps.append(f"- {cleaned_line}")
                    
                    # Replace the section
                    fixed_output = fixed_output.replace(f"Steps:{steps_text}", '\n'.join(new_steps))
        
        return fixed_output