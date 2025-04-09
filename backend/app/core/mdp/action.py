"""
Module for defining actions in the MDP framework.
"""
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import re
import copy
import random
from app.utils.logger import get_logger
from app.core.mdp.state import PromptState

logger = get_logger("mdp.action")

class Action:
    """
    Base class for actions in the MDP framework.
    
    An action represents a way to modify a prompt state.
    """
    
    def __init__(self, 
                 action_type: str,
                 description: str,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an action.
        
        Args:
            action_type: Type of the action.
            description: Human-readable description of the action.
            parameters: Additional parameters for the action.
        """
        self.action_type = action_type
        self.description = description
        self.parameters = parameters or {}
        
        logger.debug(f"Created action: {self.action_type} - {self.description}")
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to a state to generate a new state.
        
        This is an abstract method that should be implemented by subclasses.
        
        Args:
            state: Current state.
            
        Returns:
            New state after applying the action.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        
        # Add implementation in subclasses
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Default implementation
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert action to a dictionary representation.
        
        Returns:
            Dictionary representation of the action.
        """
        return {
            "action_type": self.action_type,
            "description": self.description,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """
        Create an action from a dictionary representation.
        
        Args:
            data: Dictionary with action data.
            
        Returns:
            New Action instance.
        """
        action_type = data["action_type"]
        
        # Select the appropriate action subclass based on type
        action_classes = {
            # Structural actions
            "add_role": AddRoleAction,
            "add_goal": AddGoalAction,
            "modify_workflow": ModifyWorkflowAction,
            "add_constraint": AddConstraintAction,
            
            # Content actions
            "add_explanation": AddExplanationAction,
            "add_example": AddExampleAction,
            "adjust_detail": AdjustDetailAction,
            
            # Knowledge actions
            "add_domain_knowledge": AddDomainKnowledgeAction,
            "clarify_terminology": ClarifyTerminologyAction,
            "add_rule": AddRuleAction,
            
            # Format actions
            "specify_format": SpecifyFormatAction,
            "add_template": AddTemplateAction,
            "structure_output": StructureOutputAction
        }
        
        if action_type not in action_classes:
            # Default to base class if type not recognized
            return cls(
                action_type=data["action_type"],
                description=data["description"],
                parameters=data["parameters"]
            )
        
        # Create instance of the appropriate subclass
        return action_classes[action_type](
            description=data["description"],
            parameters=data["parameters"]
        )
    
    def __str__(self) -> str:
        """
        Get string representation of the action.
        
        Returns:
            String representation.
        """
        return f"{self.action_type}: {self.description}"
    
    def __repr__(self) -> str:
        """
        Get detailed string representation of the action.
        
        Returns:
            Detailed string representation.
        """
        return f"Action({self.action_type}, {self.description})"


class StructuralAction(Action):
    """Base class for actions that modify the prompt's structure."""
    
    def __init__(self, action_type: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(f"structural.{action_type}", description, parameters)


class AddRoleAction(StructuralAction):
    """Action to add or modify a role in the prompt."""
    
    def __init__(self, description: str = "Add or modify role", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddRoleAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - role_text: Text describing the role.
                - replace: Whether to replace existing role (default: True).
        """
        params = parameters or {}
        if "role_text" not in params:
            raise ValueError("role_text parameter is required for AddRoleAction")
            
        super().__init__("add_role", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add or modify the role in the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the role added or modified.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        role_text = self.parameters["role_text"]
        replace = self.parameters.get("replace", True)
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        if components.get("role") and not replace:
            # Append to existing role if replace is False
            components["role"] = f"{components['role']}. {role_text}"
        else:
            # Replace or set new role
            components["role"] = role_text
        
        # Modify the prompt text
        text = new_state.text
        
        # Check if there's already a role section
        role_pattern = re.compile(r"(?i)role\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
        match = role_pattern.search(text)
        
        if match:
            if replace:
                # Replace existing role section
                text = text[:match.start(1)] + role_text + text[match.end(1):]
            else:
                # Append to existing role section
                text = text[:match.end(1)] + ". " + role_text + text[match.end(1):]
        else:
            # Add new role section at the beginning
            if text.strip():
                text = f"Role: {role_text}\n\n{text}"
            else:
                text = f"Role: {role_text}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the action would make a meaningful change
        replace = self.parameters.get("replace", True)
        existing_role = state.components.get("role")
        
        if existing_role is None:
            return True  # No role yet, so definitely applicable
        
        if replace:
            # Only applicable if role_text is different from existing role
            return self.parameters["role_text"] != existing_role
        else:
            # If not replacing, check if the new role text is already included
            role_text = self.parameters["role_text"]
            return role_text not in existing_role


class AddGoalAction(StructuralAction):
    """Action to add or modify a task/goal in the prompt."""
    
    def __init__(self, description: str = "Add or modify task goal", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddGoalAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - goal_text: Text describing the task goal.
                - replace: Whether to replace existing goal (default: True).
        """
        params = parameters or {}
        if "goal_text" not in params:
            raise ValueError("goal_text parameter is required for AddGoalAction")
            
        super().__init__("add_goal", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add or modify the task goal in the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the task goal added or modified.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        goal_text = self.parameters["goal_text"]
        replace = self.parameters.get("replace", True)
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        if components.get("task") and not replace:
            # Append to existing task if replace is False
            components["task"] = f"{components['task']}. {goal_text}"
        else:
            # Replace or set new task
            components["task"] = goal_text
        
        # Modify the prompt text
        text = new_state.text
        
        # Check if there's already a task section
        task_pattern = re.compile(r"(?i)task\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
        match = task_pattern.search(text)
        
        if match:
            if replace:
                # Replace existing task section
                text = text[:match.start(1)] + goal_text + text[match.end(1):]
            else:
                # Append to existing task section
                text = text[:match.end(1)] + ". " + goal_text + text[match.end(1):]
        else:
            # Add new task section after role or at the beginning
            role_pattern = re.compile(r"(?i)role\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            role_match = role_pattern.search(text)
            
            if role_match:
                # Add after role
                text = text[:role_match.end()] + f"\n\nTask: {goal_text}" + text[role_match.end():]
            else:
                # Add at the beginning
                if text.strip():
                    text = f"Task: {goal_text}\n\n{text}"
                else:
                    text = f"Task: {goal_text}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the action would make a meaningful change
        replace = self.parameters.get("replace", True)
        existing_task = state.components.get("task")
        
        if replace or not existing_task:
            return True
        
        # If not replacing, check if the new goal text is already included
        goal_text = self.parameters["goal_text"]
        return goal_text not in existing_task


class ModifyWorkflowAction(StructuralAction):
    """Action to modify the workflow steps in the prompt."""
    
    def __init__(self, description: str = "Modify workflow steps", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a ModifyWorkflowAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - steps: List of step texts or single step text to add.
                - operation: One of 'replace', 'add', 'remove', 'reorder' (default: 'replace').
                - indices: Indices for targeted operations (for add/remove/reorder).
        """
        params = parameters or {}
        if "steps" not in params:
            raise ValueError("steps parameter is required for ModifyWorkflowAction")
            
        super().__init__("modify_workflow", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to modify the workflow steps in the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the workflow steps modified.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        steps = self.parameters["steps"]
        operation = self.parameters.get("operation", "replace")
        indices = self.parameters.get("indices", [])
        
        # Convert single step to list if needed
        if isinstance(steps, str):
            steps = [steps]
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        existing_steps = components.get("steps", [])
        
        # Apply operation to steps
        if operation == "replace":
            components["steps"] = steps
        elif operation == "add":
            if not indices:
                # Add to the end if no indices specified
                components["steps"] = existing_steps + steps
            else:
                # Add at specified indices
                new_steps = existing_steps.copy()
                for i, step in zip(indices, steps):
                    if i < 0:
                        i = max(0, len(new_steps) + i + 1)
                    else:
                        i = min(len(new_steps), i)
                    new_steps.insert(i, step)
                components["steps"] = new_steps
        elif operation == "remove":
            if not indices:
                # Remove by content if no indices specified
                new_steps = [s for s in existing_steps if s not in steps]
                components["steps"] = new_steps
            else:
                # Remove at specified indices
                new_steps = existing_steps.copy()
                for i in sorted(indices, reverse=True):
                    if 0 <= i < len(new_steps):
                        new_steps.pop(i)
                components["steps"] = new_steps
        elif operation == "reorder":
            if indices and len(indices) == len(existing_steps):
                # Reorder according to indices
                new_steps = [existing_steps[i] for i in indices if 0 <= i < len(existing_steps)]
                components["steps"] = new_steps
            else:
                # Random shuffle if indices not specified correctly
                new_steps = existing_steps.copy()
                random.shuffle(new_steps)
                components["steps"] = new_steps
        
        # Modify the prompt text
        text = new_state.text
        
        # Check if there's already a steps section
        steps_pattern = re.compile(r"(?i)steps\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
        match = steps_pattern.search(text)
        
        formatted_steps = "\n".join([f"- {step}" for step in components["steps"]])
        
        if match:
            # Replace existing steps section
            text = text[:match.start(1)] + f"\n{formatted_steps}" + text[match.end(1):]
        else:
            # Add new steps section after task or role or at the beginning
            sections_pattern = re.compile(r"(?i)(task|role)\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            sections = list(sections_pattern.finditer(text))
            
            if sections:
                # Add after the last section found
                last_section = sections[-1]
                text = text[:last_section.end()] + f"\n\nSteps:\n{formatted_steps}" + text[last_section.end():]
            else:
                # Add at the beginning
                if text.strip():
                    text = f"Steps:\n{formatted_steps}\n\n{text}"
                else:
                    text = f"Steps:\n{formatted_steps}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        operation = self.parameters.get("operation", "replace")
        existing_steps = state.components.get("steps", [])
        
        if operation == "replace":
            # Always applicable for replace
            return True
        elif operation == "add":
            # Always applicable for add
            return True
        elif operation == "remove":
            # Only applicable if there are steps to remove
            return bool(existing_steps)
        elif operation == "reorder":
            # Only applicable if there are multiple steps to reorder
            return len(existing_steps) > 1
        
        return False


class AddConstraintAction(StructuralAction):
    """Action to add constraints or considerations to the prompt."""
    
    def __init__(self, description: str = "Add constraint or consideration", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddConstraintAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - constraint_text: Text describing the constraint.
                - location: Where to add the constraint ('after_role', 'after_task', 
                           'after_steps', 'end', or 'separate_section') (default: 'end').
        """
        params = parameters or {}
        if "constraint_text" not in params:
            raise ValueError("constraint_text parameter is required for AddConstraintAction")
            
        super().__init__("add_constraint", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add constraints to the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the constraints added.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        constraint_text = self.parameters["constraint_text"]
        location = self.parameters.get("location", "end")
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Add constraint to components if it's not already there
        if "constraints" not in components:
            components["constraints"] = []
        
        if isinstance(components["constraints"], list):
            components["constraints"].append(constraint_text)
        else:
            # If constraints is not a list, convert it
            components["constraints"] = [components["constraints"], constraint_text]
        
        # Modify the prompt text
        text = new_state.text
        
        # Format constraint text
        formatted_constraint = f"Constraints: {constraint_text}"
        
        # Add constraint based on location
        if location == "after_role":
            role_pattern = re.compile(r"(?i)role\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = role_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_constraint}" + text[match.end():]
            else:
                # Fall back to end if role not found
                text = text.rstrip() + f"\n\n{formatted_constraint}"
                
        elif location == "after_task":
            task_pattern = re.compile(r"(?i)task\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = task_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_constraint}" + text[match.end():]
            else:
                # Fall back to end if task not found
                text = text.rstrip() + f"\n\n{formatted_constraint}"
                
        elif location == "after_steps":
            steps_pattern = re.compile(r"(?i)steps\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = steps_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_constraint}" + text[match.end():]
            else:
                # Fall back to end if steps not found
                text = text.rstrip() + f"\n\n{formatted_constraint}"
                
        elif location == "separate_section":
            # Check if there's already a constraints section
            constraints_pattern = re.compile(r"(?i)constraints\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = constraints_pattern.search(text)
            
            if match:
                # Append to existing constraints section
                text = text[:match.end()] + f"\n- {constraint_text}" + text[match.end():]
            else:
                # Add new constraints section at the end
                text = text.rstrip() + f"\n\nConstraints:\n- {constraint_text}"
                
        else:  # Default to end
            text = text.rstrip() + f"\n\n{formatted_constraint}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the constraint is already in the prompt
        constraint_text = self.parameters["constraint_text"]
        
        if "constraints" in state.components:
            constraints = state.components["constraints"]
            if isinstance(constraints, list):
                return constraint_text not in constraints
            else:
                return constraint_text not in str(constraints)
        
        # No constraints yet, so definitely applicable
        return True


class ContentAction(Action):
    """Base class for actions that enhance the informational content of a prompt."""
    
    def __init__(self, action_type: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(f"content.{action_type}", description, parameters)


class AddExplanationAction(ContentAction):
    """Action to add explanations or clarifications to the prompt."""
    
    def __init__(self, description: str = "Add explanation or clarification", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddExplanationAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - explanation_text: Text of the explanation.
                - target: What the explanation is about ('task', 'step', 'concept', etc.).
                - target_index: Index for targeted explanations (e.g., for specific step).
        """
        params = parameters or {}
        if "explanation_text" not in params:
            raise ValueError("explanation_text parameter is required for AddExplanationAction")
            
        super().__init__("add_explanation", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add an explanation to the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the explanation added.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        explanation_text = self.parameters["explanation_text"]
        target = self.parameters.get("target", "task")
        target_index = self.parameters.get("target_index", None)
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Track if explanation has been added to components
        explanation_added = False
        
        # Add explanation to appropriate component based on target
        if target == "task" and "task" in components:
            components["task"] = f"{components['task']} ({explanation_text})"
            explanation_added = True
        elif target == "step" and "steps" in components and components["steps"]:
            steps = components["steps"]
            if target_index is not None and 0 <= target_index < len(steps):
                steps[target_index] = f"{steps[target_index]} ({explanation_text})"
                explanation_added = True
            elif steps:
                # Add to the last step if no index specified
                steps[-1] = f"{steps[-1]} ({explanation_text})"
                explanation_added = True
        
        # If not added to components, add to explanations section
        if not explanation_added:
            if "explanations" not in components:
                components["explanations"] = []
            
            if isinstance(components["explanations"], list):
                components["explanations"].append(explanation_text)
            else:
                components["explanations"] = [components["explanations"], explanation_text]
        
        # Modify the prompt text
        text = new_state.text
        
        # Different strategies based on target
        if target == "task":
            task_pattern = re.compile(r"(?i)task\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = task_pattern.search(text)
            
            if match and explanation_added:
                # Explanation already added to task component
                text = text[:match.start(1)] + components["task"] + text[match.end(1):]
            else:
                # Add as a note after task
                task_pattern = re.compile(r"(?i)task\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
                match = task_pattern.search(text)
                
                if match:
                    text = text[:match.end()] + f"\n\nNote: {explanation_text}" + text[match.end():]
                else:
                    # No task found, add at the end
                    text = text.rstrip() + f"\n\nNote: {explanation_text}"
        
        elif target == "step":
            steps_pattern = re.compile(r"(?i)steps\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = steps_pattern.search(text)
            
            if match and explanation_added:
                # Format steps with the modified step
                formatted_steps = "\n".join([f"- {step}" for step in components["steps"]])
                text = text[:match.start(1)] + f"\n{formatted_steps}" + text[match.end(1):]
            else:
                # Add explanation as a note after steps
                steps_pattern = re.compile(r"(?i)steps\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
                match = steps_pattern.search(text)
                
                if match:
                    text = text[:match.end()] + f"\n\nNote: {explanation_text}" + text[match.end():]
                else:
                    # No steps found, add at the end
                    text = text.rstrip() + f"\n\nNote: {explanation_text}"
        
        else:  # Default for other targets or if components weren't updated
            # Check if there's already an explanations or notes section
            notes_pattern = re.compile(r"(?i)(explanations|notes)\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = notes_pattern.search(text)
            
            if match:
                # Append to existing notes section
                text = text[:match.end(2)] + f"\n- {explanation_text}" + text[match.end(2):]
            else:
                # Add new notes section at the end
                text = text.rstrip() + f"\n\nNotes:\n- {explanation_text}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if explanation is already in the prompt
        explanation_text = self.parameters["explanation_text"]
        target = self.parameters.get("target", "task")
        
        if target == "task" and "task" in state.components:
            # 检查 task 是否为 None 或空值
            task_content = state.components["task"]
            if task_content is None:
                return True  # 如果 task 是 None，那么解释肯定不在其中，可以应用
            return explanation_text not in task_content  # 否则检查解释是否已存在
        
        elif target == "step" and "steps" in state.components and state.components["steps"]:
            target_index = self.parameters.get("target_index", None)
            steps = state.components["steps"]
            
            if target_index is not None and 0 <= target_index < len(steps):
                # 检查步骤是否为 None
                step_content = steps[target_index]
                if step_content is None:
                    return True
                return explanation_text not in step_content
            else:
                # 检查解释是否在任何步骤中
                return not any(explanation_text in step for step in steps if step is not None)
        
        elif "explanations" in state.components:
            explanations = state.components["explanations"]
            if explanations is None:
                return True
            if isinstance(explanations, list):
                return explanation_text not in explanations
            else:
                return explanation_text not in str(explanations)
        
        # Explanation not found in any relevant component, so applicable
        return True


class AddExampleAction(ContentAction):
    """Action to add examples to the prompt."""
    
    def __init__(self, description: str = "Add example", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddExampleAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - example_text: Text of the example.
                - example_type: Type of example ('input_output', 'step_by_step', 'counterexample').
                - location: Where to add the example ('after_task', 'after_steps', 'end',
                            or 'separate_section') (default: 'separate_section').
        """
        params = parameters or {}
        if "example_text" not in params:
            raise ValueError("example_text parameter is required for AddExampleAction")
            
        super().__init__("add_example", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add an example to the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the example added.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        example_text = self.parameters["example_text"]
        example_type = self.parameters.get("example_type", "input_output")
        location = self.parameters.get("location", "separate_section")
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Add example to components
        if "examples" not in components:
            components["examples"] = []
        
        if isinstance(components["examples"], list):
            components["examples"].append({
                "text": example_text,
                "type": example_type
            })
        else:
            # If examples is not a list, convert it
            components["examples"] = [
                components["examples"],
                {"text": example_text, "type": example_type}
            ]
        
        # Modify the prompt text
        text = new_state.text
        
        # Format example based on its type
        if example_type == "input_output":
            formatted_example = f"Example:\n{example_text}"
        elif example_type == "step_by_step":
            formatted_example = f"Step-by-step example:\n{example_text}"
        elif example_type == "counterexample":
            formatted_example = f"Counterexample (what to avoid):\n{example_text}"
        else:
            formatted_example = f"Example ({example_type}):\n{example_text}"
        
        # Add example based on location
        if location == "after_task":
            task_pattern = re.compile(r"(?i)task\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = task_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_example}" + text[match.end():]
            else:
                # Fall back to end if task not found
                text = text.rstrip() + f"\n\n{formatted_example}"
                
        elif location == "after_steps":
            steps_pattern = re.compile(r"(?i)steps\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = steps_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_example}" + text[match.end():]
            else:
                # Fall back to end if steps not found
                text = text.rstrip() + f"\n\n{formatted_example}"
                
        elif location == "separate_section":
            # Check if there's already an examples section
            examples_pattern = re.compile(r"(?i)examples?\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = examples_pattern.search(text)
            
            if match:
                # Append to existing examples section
                text = text[:match.end(1)] + f"\n\n{example_text}" + text[match.end(1):]
            else:
                # Add new examples section at the end - use singular Example
                if len(components["examples"]) <= 1:
                    text = text.rstrip() + f"\n\nExample:\n{example_text}"
                else:
                    text = text.rstrip() + f"\n\nExamples:\n{example_text}"
                
        else:  # Default to end
            text = text.rstrip() + f"\n\n{formatted_example}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if example is already in the prompt
        example_text = self.parameters["example_text"]
        
        if "examples" in state.components:
            examples = state.components["examples"]
            if isinstance(examples, list):
                for example in examples:
                    if isinstance(example, dict) and example.get("text") == example_text:
                        return False
                    elif example == example_text:
                        return False
            else:
                return example_text not in str(examples)
        
        # Also check if the example text appears in the prompt text
        # This is a simple check that might have false positives
        if example_text in state.text:
            return False
        
        # Example not found, so applicable
        return True


class AdjustDetailAction(ContentAction):
    """Action to adjust the level of detail in the prompt."""
    
    def __init__(self, description: str = "Adjust level of detail", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AdjustDetailAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - direction: 'increase' or 'decrease' the level of detail.
                - target: What to adjust ('steps', 'task', 'all').
                - adjustment_text: Optional text to add or remove.
        """
        params = parameters or {}
        if "direction" not in params:
            raise ValueError("direction parameter is required for AdjustDetailAction")
            
        super().__init__("adjust_detail", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to adjust the detail level in the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the detail level adjusted.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        direction = self.parameters["direction"]
        target = self.parameters.get("target", "all")
        adjustment_text = self.parameters.get("adjustment_text", None)
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Modify the prompt text
        text = new_state.text
        
        if direction == "increase" and adjustment_text:
            # Add more detail
            if target == "steps" or target == "all":
                # Add detail to steps
                if "steps" in components and components["steps"]:
                    # Add detail to each step
                    for i in range(len(components["steps"])):
                        if not components["steps"][i].endswith(adjustment_text):
                            components["steps"][i] += f" {adjustment_text}"
                    
                    # Update steps in the text
                    steps_pattern = re.compile(r"(?i)steps\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
                    match = steps_pattern.search(text)
                    
                    if match:
                        formatted_steps = "\n".join([f"- {step}" for step in components["steps"]])
                        text = text[:match.start(1)] + f"\n{formatted_steps}" + text[match.end(1):]
            
            if target == "task" or target == "all":
                # Add detail to task
                if "task" in components and components["task"]:
                    if not components["task"].endswith(adjustment_text):
                        components["task"] += f" {adjustment_text}"
                    
                    # Update task in the text
                    task_pattern = re.compile(r"(?i)task\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
                    match = task_pattern.search(text)
                    
                    if match:
                        text = text[:match.start(1)] + components["task"] + text[match.end(1):]
                        
        elif direction == "decrease":
            # Simplify by removing unnecessary detail
            if target == "steps" or target == "all":
                # Simplify steps
                if "steps" in components and components["steps"]:
                    for i in range(len(components["steps"])):
                        # Remove adjustment text if specified
                        if adjustment_text and adjustment_text in components["steps"][i]:
                            components["steps"][i] = components["steps"][i].replace(adjustment_text, "").strip()
                        # Otherwise, simplify by taking first sentence
                        elif "." in components["steps"][i]:
                            components["steps"][i] = components["steps"][i].split(".")[0] + "."
                    
                    # Update steps in the text
                    steps_pattern = re.compile(r"(?i)steps\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
                    match = steps_pattern.search(text)
                    
                    if match:
                        formatted_steps = "\n".join([f"- {step}" for step in components["steps"]])
                        text = text[:match.start(1)] + f"\n{formatted_steps}" + text[match.end(1):]
            
            if target == "task" or target == "all":
                # Simplify task
                if "task" in components and components["task"]:
                    # Remove adjustment text if specified
                    if adjustment_text and adjustment_text in components["task"]:
                        components["task"] = components["task"].replace(adjustment_text, "").strip()
                    # Otherwise, simplify by taking first sentence
                    elif "." in components["task"]:
                        components["task"] = components["task"].split(".")[0] + "."
                    
                    # Update task in the text
                    task_pattern = re.compile(r"(?i)task\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
                    match = task_pattern.search(text)
                    
                    if match:
                        text = text[:match.start(1)] + components["task"] + text[match.end(1):]
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        direction = self.parameters["direction"]
        target = self.parameters.get("target", "all")
        adjustment_text = self.parameters.get("adjustment_text", None)
        
        if direction == "increase" and adjustment_text:
            # Check if the adjustment text is already present in the target components
            if target == "steps" or target == "all":
                if "steps" in state.components and state.components["steps"]:
                    # If all steps already have the adjustment text, not applicable
                    if all(adjustment_text in step for step in state.components["steps"]):
                        return False
            
            if target == "task" or target == "all":
                if "task" in state.components and state.components["task"]:
                    if adjustment_text in state.components["task"]:
                        return False
        
        elif direction == "decrease":
            # Check if there's anything to simplify
            if target == "steps" or target == "all":
                if "steps" not in state.components or not state.components["steps"]:
                    return False
                
                if adjustment_text:
                    # If adjustment text is not in any step, not applicable
                    if not any(adjustment_text in step for step in state.components["steps"]):
                        return False
            
            if target == "task" or target == "all":
                if "task" not in state.components or not state.components["task"]:
                    return False
                
                if adjustment_text and adjustment_text not in state.components["task"]:
                    return False
        
        return True


class KnowledgeAction(Action):
    """Base class for actions that incorporate domain expertise into a prompt."""
    
    def __init__(self, action_type: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(f"knowledge.{action_type}", description, parameters)


class AddDomainKnowledgeAction(KnowledgeAction):
    """Action to add domain-specific knowledge to the prompt."""
    
    def __init__(self, description: str = "Add domain knowledge", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddDomainKnowledgeAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - knowledge_text: Text describing the domain knowledge.
                - domain: Domain the knowledge belongs to (e.g., 'medical', 'legal').
                - location: Where to add the knowledge ('after_role', 'after_task', 
                           'separate_section', 'before_steps') (default: 'separate_section').
        """
        params = parameters or {}
        if "knowledge_text" not in params:
            raise ValueError("knowledge_text parameter is required for AddDomainKnowledgeAction")
            
        super().__init__("add_domain_knowledge", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add domain knowledge to the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the domain knowledge added.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        knowledge_text = self.parameters["knowledge_text"]
        domain = self.parameters.get("domain", "")
        location = self.parameters.get("location", "separate_section")
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Add knowledge to components
        if "domain_knowledge" not in components:
            components["domain_knowledge"] = []
        
        domain_entry = {"text": knowledge_text}
        if domain:
            domain_entry["domain"] = domain
            
        if isinstance(components["domain_knowledge"], list):
            components["domain_knowledge"].append(domain_entry)
        else:
            # If domain_knowledge is not a list, convert it
            components["domain_knowledge"] = [
                components["domain_knowledge"],
                domain_entry
            ]
        
        # Modify the prompt text
        text = new_state.text
        
        # Format knowledge text
        if domain:
            formatted_knowledge = f"Domain Knowledge ({domain}): {knowledge_text}"
        else:
            formatted_knowledge = f"Domain Knowledge: {knowledge_text}"
        
        # Add knowledge based on location
        if location == "after_role":
            role_pattern = re.compile(r"(?i)role\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = role_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_knowledge}" + text[match.end():]
            else:
                # Fall back to separate section if role not found
                location = "separate_section"
                
        if location == "after_task":
            task_pattern = re.compile(r"(?i)task\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = task_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_knowledge}" + text[match.end():]
            else:
                # Fall back to separate section if task not found
                location = "separate_section"
                
        if location == "before_steps":
            steps_pattern = re.compile(r"(?i)steps\s*:", re.DOTALL)
            match = steps_pattern.search(text)
            
            if match:
                text = text[:match.start()] + f"{formatted_knowledge}\n\n" + text[match.start():]
            else:
                # Fall back to separate section if steps not found
                location = "separate_section"
                
        if location == "separate_section":
            # Check if there's already a domain knowledge section
            knowledge_pattern = re.compile(r"(?i)domain knowledge\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = knowledge_pattern.search(text)
            
            if match:
                # Append to existing domain knowledge section
                if domain:
                    text = text[:match.end(1)] + f"\n- ({domain}): {knowledge_text}" + text[match.end(1):]
                else:
                    text = text[:match.end(1)] + f"\n- {knowledge_text}" + text[match.end(1):]
            else:
                # Add new domain knowledge section after task and before steps
                task_pattern = re.compile(r"(?i)task\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
                steps_pattern = re.compile(r"(?i)steps\s*:", re.DOTALL)
                task_match = task_pattern.search(text)
                steps_match = steps_pattern.search(text)
                
                if task_match and steps_match and task_match.end() < steps_match.start():
                    # Insert between task and steps
                    if domain:
                        text = (text[:task_match.end()] + 
                               f"\n\nDomain Knowledge ({domain}):\n- {knowledge_text}" + 
                               text[task_match.end():])
                    else:
                        text = (text[:task_match.end()] + 
                               f"\n\nDomain Knowledge:\n- {knowledge_text}" + 
                               text[task_match.end():])
                else:
                    # Add at the end
                    if domain:
                        text = text.rstrip() + f"\n\nDomain Knowledge ({domain}):\n- {knowledge_text}"
                    else:
                        text = text.rstrip() + f"\n\nDomain Knowledge:\n- {knowledge_text}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the knowledge is already in the prompt
        knowledge_text = self.parameters["knowledge_text"]
        
        if "domain_knowledge" in state.components:
            domain_knowledge = state.components["domain_knowledge"]
            if isinstance(domain_knowledge, list):
                for entry in domain_knowledge:
                    if isinstance(entry, dict) and entry.get("text") == knowledge_text:
                        return False
                    elif entry == knowledge_text:
                        return False
            else:
                return knowledge_text not in str(domain_knowledge)
        
        # Also check if the knowledge text appears in the prompt text
        if knowledge_text in state.text:
            return False
        
        # Knowledge not found, so applicable
        return True


class ClarifyTerminologyAction(KnowledgeAction):
    """Action to clarify specialized terminology in the prompt."""
    
    def __init__(self, description: str = "Clarify terminology", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a ClarifyTerminologyAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - term: The term to clarify.
                - definition: The definition or explanation of the term.
                - location: Where to add the clarification ('inline', 'after_first_use',
                           'separate_section') (default: 'inline').
        """
        params = parameters or {}
        if "term" not in params or "definition" not in params:
            raise ValueError("term and definition parameters are required for ClarifyTerminologyAction")
            
        super().__init__("clarify_terminology", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to clarify terminology in the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the terminology clarified.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        term = self.parameters["term"]
        definition = self.parameters["definition"]
        location = self.parameters.get("location", "inline")
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Add terminology to components
        if "terminology" not in components:
            components["terminology"] = {}
        
        if isinstance(components["terminology"], dict):
            components["terminology"][term] = definition
        else:
            # If terminology is not a dict, convert it
            components["terminology"] = {term: definition}
        
        # Modify the prompt text
        text = new_state.text
        
        if location == "inline":
            # Replace term with term (definition)
            # Use word boundaries to avoid partial word matches
            term_pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
            # Only replace the first occurrence
            match = term_pattern.search(text)
            if match:
                replacement = f"{match.group(0)} ({definition})"
                text = text[:match.start()] + replacement + text[match.end():]
            else:
                # Term not found, add as a separate section
                location = "separate_section"
                
        elif location == "after_first_use":
            # Add definition after the first use of the term
            term_pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
            # Find the first occurrence after paragraphs
            match = term_pattern.search(text)
            if match:
                # Find the end of the sentence or paragraph
                end_of_sentence = text.find(".", match.end())
                end_of_paragraph = text.find("\n", match.end())
                
                if end_of_sentence != -1 and (end_of_paragraph == -1 or end_of_sentence < end_of_paragraph):
                    insertion_point = end_of_sentence + 1
                elif end_of_paragraph != -1:
                    insertion_point = end_of_paragraph
                else:
                    insertion_point = len(text)
                
                # Insert definition
                text = text[:insertion_point] + f" [{term}: {definition}]" + text[insertion_point:]
            else:
                # Term not found, add as a separate section
                location = "separate_section"
        
        if location == "separate_section":
            # Check if there's already a terminology section
            terminology_pattern = re.compile(r"(?i)(terminology|definitions)\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = terminology_pattern.search(text)
            
            if match:
                # Append to existing terminology section
                text = text[:match.end(2)] + f"\n- {term}: {definition}" + text[match.end(2):]
            else:
                # Add new terminology section at the end
                text = text.rstrip() + f"\n\nTerminology:\n- {term}: {definition}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the term is already defined in the prompt
        term = self.parameters["term"]
        definition = self.parameters["definition"]
        
        if "terminology" in state.components:
            terminology = state.components["terminology"]
            if isinstance(terminology, dict) and term in terminology:
                return False
            elif isinstance(terminology, str) and term in terminology:
                return False
        
        # Check if the term and definition pair appears in the text
        if f"{term} ({definition})" in state.text or f"{term}: {definition}" in state.text:
            return False
        
        # Term not defined, so applicable
        return True


class AddRuleAction(KnowledgeAction):
    """Action to add domain-specific rules to the prompt."""
    
    def __init__(self, description: str = "Add domain-specific rule", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddRuleAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - rule_text: Text describing the rule.
                - priority: Priority of the rule ('high', 'medium', 'low').
                - location: Where to add the rule ('after_steps', 'separate_section',
                           'constraints') (default: 'separate_section').
        """
        params = parameters or {}
        if "rule_text" not in params:
            raise ValueError("rule_text parameter is required for AddRuleAction")
            
        super().__init__("add_rule", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add a rule to the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the rule added.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        rule_text = self.parameters["rule_text"]
        priority = self.parameters.get("priority", "medium")
        location = self.parameters.get("location", "separate_section")
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Add rule to components
        if "rules" not in components:
            components["rules"] = []
        
        rule_entry = {"text": rule_text, "priority": priority}
            
        if isinstance(components["rules"], list):
            components["rules"].append(rule_entry)
        else:
            # If rules is not a list, convert it
            components["rules"] = [components["rules"], rule_entry]
        
        # Modify the prompt text
        text = new_state.text
        
        # Format rule text based on priority
        if priority == "high":
            formatted_rule = f"IMPORTANT RULE: {rule_text}"
        elif priority == "medium":
            formatted_rule = f"Rule: {rule_text}"
        else:  # low
            formatted_rule = f"Guideline: {rule_text}"
        
        if location == "after_steps":
            steps_pattern = re.compile(r"(?i)steps\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = steps_pattern.search(text)
            
            if match:
                text = text[:match.end()] + f"\n\n{formatted_rule}" + text[match.end():]
            else:
                # Fall back to separate section if steps not found
                location = "separate_section"
                
        elif location == "constraints":
            # Check if there's already a constraints section
            constraints_pattern = re.compile(r"(?i)constraints\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = constraints_pattern.search(text)
            
            if match:
                # Append to existing constraints section
                text = text[:match.end(1)] + f"\n- {rule_text}" + text[match.end(1):]
            else:
                # Add new constraints section after steps or at the end
                steps_pattern = re.compile(r"(?i)steps\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
                steps_match = steps_pattern.search(text)
                
                if steps_match:
                    text = text[:steps_match.end()] + f"\n\nConstraints:\n- {rule_text}" + text[steps_match.end():]
                else:
                    text = text.rstrip() + f"\n\nConstraints:\n- {rule_text}"
                    
        elif location == "separate_section":
            # Check if there's already a rules section
            rules_pattern = re.compile(r"(?i)(rules|guidelines)\s*:(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
            match = rules_pattern.search(text)
            
            if match:
                # Append to existing rules section
                if priority == "high":
                    text = text[:match.end(2)] + f"\n- IMPORTANT: {rule_text}" + text[match.end(2):]
                else:
                    text = text[:match.end(2)] + f"\n- {rule_text}" + text[match.end(2):]
            else:
                # Add new rules section after steps or at the end
                steps_pattern = re.compile(r"(?i)steps\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
                steps_match = steps_pattern.search(text)
                
                if steps_match:
                    text = text[:steps_match.end()] + f"\n\nRules:\n- {rule_text}" + text[steps_match.end():]
                else:
                    text = text.rstrip() + f"\n\nRules:\n- {rule_text}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the rule is already in the prompt
        rule_text = self.parameters["rule_text"]
        
        if "rules" in state.components:
            rules = state.components["rules"]
            if isinstance(rules, list):
                for entry in rules:
                    if isinstance(entry, dict) and entry.get("text") == rule_text:
                        return False
                    elif entry == rule_text:
                        return False
            else:
                return rule_text not in str(rules)
        
        # Also check if the rule text appears in the prompt text
        if rule_text in state.text:
            return False
        
        # Rule not found, so applicable
        return True


class FormatAction(Action):
    """Base class for actions that optimize output structure of a prompt."""
    
    def __init__(self, action_type: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(f"format.{action_type}", description, parameters)


class SpecifyFormatAction(FormatAction):
    """Action to specify output format in the prompt."""
    
    def __init__(self, description: str = "Specify output format", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a SpecifyFormatAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - format_text: Text describing the output format.
                - replace: Whether to replace existing format (default: True).
        """
        params = parameters or {}
        if "format_text" not in params:
            raise ValueError("format_text parameter is required for SpecifyFormatAction")
            
        super().__init__("specify_format", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to specify output format in the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the output format specified.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        format_text = self.parameters["format_text"]
        replace = self.parameters.get("replace", True)
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        if components.get("output_format") and not replace:
            # Append to existing format if replace is False
            components["output_format"] = f"{components['output_format']}. {format_text}"
        else:
            # Replace or set new format
            components["output_format"] = format_text
        
        # Modify the prompt text
        text = new_state.text
        
        # Check if there's already an output format section
        format_pattern = re.compile(r"(?i)output\s+format\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)", re.DOTALL)
        match = format_pattern.search(text)
        
        if match:
            if replace:
                # Replace existing format section
                text = text[:match.start(1)] + format_text + text[match.end(1):]
            else:
                # Append to existing format section
                text = text[:match.end(1)] + ". " + format_text + text[match.end(1):]
        else:
            # Add new format section at the end
            text = text.rstrip() + f"\n\nOutput Format: {format_text}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the format would make a meaningful change
        replace = self.parameters.get("replace", True)
        existing_format = state.components.get("output_format")
        format_text = self.parameters["format_text"]
        
        if replace or not existing_format:
            return True
        
        # If not replacing, check if the new format text is already included
        return format_text not in existing_format


class AddTemplateAction(FormatAction):
    """Action to add output templates to the prompt."""
    
    def __init__(self, description: str = "Add output template", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize an AddTemplateAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - template_text: Text of the template.
                - template_type: Type of template ('example', 'structure', 'schema').
        """
        params = parameters or {}
        if "template_text" not in params:
            raise ValueError("template_text parameter is required for AddTemplateAction")
            
        super().__init__("add_template", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add an output template to the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the output template added.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        template_text = self.parameters["template_text"]
        template_type = self.parameters.get("template_type", "example")
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Add template to components
        if "templates" not in components:
            components["templates"] = []
        
        template_entry = {"text": template_text, "type": template_type}
            
        if isinstance(components["templates"], list):
            components["templates"].append(template_entry)
        else:
            # If templates is not a list, convert it
            components["templates"] = [components["templates"], template_entry]
        
        # Modify the prompt text
        text = new_state.text
        
        # Format template based on its type
        if template_type == "example":
            formatted_template = f"Output Example:\n{template_text}"
        elif template_type == "structure":
            formatted_template = f"Output Structure:\n{template_text}"
        elif template_type == "schema":
            formatted_template = f"Output Schema:\n{template_text}"
        else:
            formatted_template = f"Output Template ({template_type}):\n{template_text}"
        
        # Add template after output format or at the end
        format_pattern = re.compile(r"(?i)output\s+format\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
        format_match = format_pattern.search(text)
        
        if format_match:
            text = text[:format_match.end()] + f"\n\n{formatted_template}" + text[format_match.end():]
        else:
            # Add at the end
            text = text.rstrip() + f"\n\n{formatted_template}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if the template is already in the prompt
        template_text = self.parameters["template_text"]
        
        if "templates" in state.components:
            templates = state.components["templates"]
            if isinstance(templates, list):
                for entry in templates:
                    if isinstance(entry, dict) and entry.get("text") == template_text:
                        return False
                    elif entry == template_text:
                        return False
            else:
                return template_text not in str(templates)
        
        # Also check if the template text appears in the prompt text
        if template_text in state.text:
            return False
        
        # Template not found, so applicable
        return True


class StructureOutputAction(FormatAction):
    """Action to add structured output organization to the prompt."""
    
    def __init__(self, description: str = "Structure output organization", 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a StructureOutputAction.
        
        Args:
            description: Description of the action.
            parameters: Dictionary with keys:
                - structure_type: Type of structure ('headings', 'numbered', 'bullets', 'sections').
                - elements: List of elements to include in the structure.
        """
        params = parameters or {}
        if "structure_type" not in params:
            raise ValueError("structure_type parameter is required for StructureOutputAction")
            
        super().__init__("structure_output", description, params)
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the action to add structured output organization to the prompt.
        
        Args:
            state: Current state.
            
        Returns:
            New state with the structured output organization added.
        """
        # Create a copy of the state to modify
        new_state = state.copy()
        structure_type = self.parameters["structure_type"]
        elements = self.parameters.get("elements", [])
        
        # Update components first
        components = copy.deepcopy(new_state.components)
        
        # Add structure to components
        components["output_structure"] = {
            "type": structure_type,
            "elements": elements
        }
        
        # Modify the prompt text
        text = new_state.text
        
        # Format structure based on type and elements
        structure_intro = f"Organize the output using the following structure:"
        
        if structure_type == "headings":
            if elements:
                formatted_structure = structure_intro + "\n" + "\n".join([f"# {elem}" for elem in elements])
            else:
                formatted_structure = structure_intro + "\n# Use clear headings to organize the content"
                
        elif structure_type == "numbered":
            if elements:
                formatted_structure = structure_intro + "\n" + "\n".join([f"{i+1}. {elem}" for i, elem in enumerate(elements)])
            else:
                formatted_structure = structure_intro + "\nUse numbered points for sequential steps or hierarchical organization"
                
        elif structure_type == "bullets":
            if elements:
                formatted_structure = structure_intro + "\n" + "\n".join([f"- {elem}" for elem in elements])
            else:
                formatted_structure = structure_intro + "\nUse bullet points for non-sequential items"
                
        elif structure_type == "sections":
            if elements:
                formatted_structure = structure_intro + "\nInclude these sections:\n" + "\n".join([f"- {elem}" for elem in elements])
            else:
                formatted_structure = structure_intro + "\nDivide the output into clear sections with appropriate headings"
                
        else:
            formatted_structure = structure_intro + f"\nUse a {structure_type} structure for clarity and organization"
        
        # Add structure after output format or template or at the end
        format_pattern = re.compile(r"(?i)output\s+(format|template|example|structure|schema)\s*:.*?(?=\n\s*\w+\s*:|$)", re.DOTALL)
        matches = list(format_pattern.finditer(text))
        
        if matches:
            last_match = matches[-1]
            text = text[:last_match.end()] + f"\n\n{formatted_structure}" + text[last_match.end():]
        else:
            # Add at the end
            text = text.rstrip() + f"\n\n{formatted_structure}"
        
        # Create a new state with the modified text and components
        new_state = PromptState(
            text=text,
            components=components,
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied=str(self)
        )
        
        return new_state
    
    def is_applicable(self, state: PromptState) -> bool:
        """
        Check if the action is applicable to the given state.
        
        Args:
            state: State to check.
            
        Returns:
            True if action can be applied, False otherwise.
        """
        # Check if an output structure is already specified
        if "output_structure" in state.components:
            existing_structure = state.components["output_structure"]
            if isinstance(existing_structure, dict):
                existing_type = existing_structure.get("type")
                if existing_type == self.parameters["structure_type"]:
                    return False
        
        # No conflicting structure found, so applicable
        return True


# Create a registry of action classes for easy lookup
action_registry = {
    # Structural actions
    "add_role": AddRoleAction,
    "add_goal": AddGoalAction,
    "modify_workflow": ModifyWorkflowAction,
    "add_constraint": AddConstraintAction,
    
    # Content actions
    "add_explanation": AddExplanationAction,
    "add_example": AddExampleAction,
    "adjust_detail": AdjustDetailAction,
    
    # Knowledge actions
    "add_domain_knowledge": AddDomainKnowledgeAction,
    "clarify_terminology": ClarifyTerminologyAction,
    "add_rule": AddRuleAction,
    
    # Format actions
    "specify_format": SpecifyFormatAction,
    "add_template": AddTemplateAction,
    "structure_output": StructureOutputAction
}

# 添加evolution动作
class MutationAction(Action):
    """Action representing a mutation operation."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a mutation action.
        
        Args:
            parameters: Dictionary with mutation parameters.
        """
        super().__init__(
            action_type="mutation",
            description="Apply mutation to prompt",
            parameters=parameters or {}
        )
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the mutation action to a state.
        
        Args:
            state: Current state.
            
        Returns:
            New state after applying mutation.
        """
        # In a real implementation, this would call a mutation function
        # For now, just return a copy of the state
        new_state = state.copy()
        new_state.text = new_state.text
        return new_state


class CrossoverAction(Action):
    """Action representing a crossover operation."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a crossover action.
        
        Args:
            parameters: Dictionary with crossover parameters.
        """
        super().__init__(
            action_type="crossover",
            description="Apply crossover to prompts",
            parameters=parameters or {}
        )
    
    def apply(self, state: PromptState) -> PromptState:
        """
        Apply the crossover action to a state.
        
        Args:
            state: Current state.
            
        Returns:
            New state after applying crossover.
        """
        # In a real implementation, this would call a crossover function
        # For now, just return a copy of the state
        new_state = state.copy()
        return new_state

# Update the action registry with the new action types
action_registry.update({
    # Evolutionary actions
    "mutation": MutationAction,
    "crossover": CrossoverAction
})


def create_action(action_type: str, **kwargs) -> Action:
    """
    Create an action of the specified type with the given parameters.
    
    Args:
        action_type: Type of action to create.
        **kwargs: Parameters for the action.
    
    Returns:
        Action instance.
    
    Raises:
        ValueError: If action_type is not recognized.
    """
    if action_type not in action_registry:
        raise ValueError(f"Unknown action type: {action_type}")
    
    return action_registry[action_type](**kwargs)