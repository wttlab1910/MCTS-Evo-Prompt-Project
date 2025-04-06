"""
Module for representing prompt states in the MDP framework.
"""
from typing import Dict, Any, List, Optional
import copy
import hashlib
import json
from app.utils.logger import get_logger

logger = get_logger("mdp.state")

class PromptState:
    """
    Represents a prompt state in the MDP framework.
    
    A state encapsulates the prompt text along with its structured 
    representation and quality metrics.
    """
    
    def __init__(
        self, 
        text: str, 
        components: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        history: Optional[List[str]] = None,
        parent: Optional['PromptState'] = None,
        action_applied: Optional[str] = None
    ):
        """
        Initialize a prompt state.
        
        Args:
            text: Complete prompt text.
            components: Structured components of the prompt (role, goal, steps, etc.).
            metrics: Quality metrics (performance, completeness, efficiency).
            history: List of actions applied to reach this state.
            parent: Parent state (if derived from another state).
            action_applied: Description of the action applied to reach this state.
        """
        self.text = text
        self.components = components or self._extract_components(text)
        self.metrics = metrics or {}
        self.history = history or []
        self.parent = parent
        self.action_applied = action_applied
        
        # If derived from a parent and action is specified, add to history
        if parent and action_applied:
            self.history = parent.history + [action_applied]
        
        # Generate a unique state identifier
        self.state_id = self._generate_id()
        
        logger.debug(f"Created state with ID: {self.state_id[:8]}")
    
    def _extract_components(self, text: str) -> Dict[str, Any]:
        """
        Extract structured components from prompt text.
        
        This is a basic implementation that looks for common components like
        Role, Task, Steps, and Output Format. Can be enhanced with more sophisticated
        parsing.
        
        Args:
            text: Prompt text.
            
        Returns:
            Dictionary of structured components.
        """
        components = {
            "role": None,
            "task": None,
            "steps": [],
            "content": None,
            "output_format": None
        }
        
        # Simple parsing based on common patterns
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if lower_line.startswith("role:"):
                current_section = "role"
                components["role"] = line[5:].strip()
            elif lower_line.startswith("task:"):
                current_section = "task"
                components["task"] = line[5:].strip()
            elif lower_line.startswith("steps:"):
                current_section = "steps"
            elif lower_line.startswith("content:"):
                current_section = "content"
                components["content"] = line[8:].strip()
            elif lower_line.startswith("output format:"):
                current_section = "output_format"
                components["output_format"] = line[14:].strip()
            # Collect step items
            elif current_section == "steps" and (line.startswith("-") or line.startswith("•")):
                components["steps"].append(line[1:].strip())
            # Append to current section if not a new section
            elif current_section:
                if current_section == "steps":
                    if components["steps"]:
                        components["steps"][-1] += " " + line
                    else:
                        components["steps"].append(line)
                else:
                    if components[current_section]:
                        components[current_section] += " " + line
                    else:
                        components[current_section] = line
        
        return components
    
    def _generate_id(self) -> str:
        """
        Generate a unique identifier for this state based on its text content.
        
        Returns:
            A unique string identifier.
        """
        # Use SHA-256 hash of the text for uniqueness
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to a dictionary representation.
        
        Returns:
            Dictionary representation of the state.
        """
        return {
            "text": self.text,
            "components": self.components,
            "metrics": self.metrics,
            "history": self.history,
            "state_id": self.state_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptState':
        """
        Create a state from a dictionary representation.
        
        Args:
            data: Dictionary with state data.
            
        Returns:
            New PromptState instance.
        """
        return cls(
            text=data["text"],
            components=data["components"],
            metrics=data["metrics"],
            history=data["history"]
        )
    
    def copy(self) -> 'PromptState':
        """
        Create a deep copy of this state.
        
        Returns:
            New PromptState instance with same content.
        """
        return PromptState(
            text=self.text,
            components=copy.deepcopy(self.components),
            metrics=copy.deepcopy(self.metrics),
            history=self.history.copy(),
            parent=self.parent,
            action_applied=self.action_applied
        )
    
    def get_depth(self) -> int:
        """
        Get the depth of this state in the state tree.
        
        Returns:
            Depth as number of actions applied.
        """
        return len(self.history)
    
    def has_component(self, component_name: str) -> bool:
        """
        Check if a specific component exists and has content.
        
        Args:
            component_name: Name of the component to check.
            
        Returns:
            True if component exists and has content, False otherwise.
        """
        if component_name not in self.components:
            return False
            
        value = self.components[component_name]
        
        if isinstance(value, str):
            return bool(value.strip())
        elif isinstance(value, list):
            return bool(value)
        else:
            return value is not None
    
    def get_structural_completeness(self) -> float:
        """
        Calculate a score representing the structural completeness of the prompt.
        
        Returns:
            Score between 0.0 and 1.0 representing completeness.
        """
        # Define required components and their weights
        required_components = {
            "role": 0.2,
            "task": 0.3,
            "steps": 0.3,
            "output_format": 0.2
        }
        
        score = 0.0
        for component, weight in required_components.items():
            if self.has_component(component):
                score += weight
        
        return score
    
    def get_token_efficiency(self) -> float:
        """
        Calculate a score representing token efficiency.
        This is a simplified implementation that can be enhanced with actual token counting.
        
        Returns:
            Score between 0.0 and 1.0 representing efficiency.
        """
        # Simple baseline implementation
        # Consider the ratio of components to total text length
        total_length = len(self.text)
        
        if total_length == 0:
            return 0.0
        
        # Calculate content density
        component_text = ""
        for component, value in self.components.items():
            if isinstance(value, str):
                component_text += value
            elif isinstance(value, list):
                # Handle list items that might be dictionaries
                for item in value:
                    if isinstance(item, dict):
                        if "text" in item:
                            component_text += item["text"]
                        else:
                            component_text += str(item)
                    else:
                        component_text += str(item)
            elif isinstance(value, dict):
                component_text += str(value)
        
        component_length = len(component_text)
        
        # Normalize to 0-1 range with a reasonable cutoff
        ratio = min(1.0, component_length / (total_length * 0.8))
        
        return ratio
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two states are equal.
        
        Args:
            other: Another state to compare with.
            
        Returns:
            True if states are equal, False otherwise.
        """
        if not isinstance(other, PromptState):
            return False
        
        return self.state_id == other.state_id
    
    def __hash__(self) -> int:
        """
        Generate a hash for the state.
        
        Returns:
            Hash value.
        """
        return hash(self.state_id)
    
    def __str__(self) -> str:
        """
        Get string representation of the state.
        
        Returns:
            String representation.
        """
        return f"PromptState(id={self.state_id[:8]}, depth={len(self.history)})"
    
    def __repr__(self) -> str:
        """
        Get detailed string representation of the state.
        
        Returns:
            Detailed string representation.
        """
        return (f"PromptState(id={self.state_id[:8]}, "
                f"depth={len(self.history)}, "
                f"metrics={self.metrics})")