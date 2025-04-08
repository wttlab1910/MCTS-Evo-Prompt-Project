"""
Knowledge integration components.

This module implements the main knowledge integration functionality,
combining knowledge items with prompts in an optimal way.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from app.utils.logger import get_logger
from app.core.mdp.state import PromptState
from app.knowledge.integration.strategy import (
    IntegrationStrategy,
    PlacementStrategy,
    FormatSelectionStrategy,
    ConflictResolutionStrategy
)

logger = get_logger("knowledge.integration")

class KnowledgeIntegrator:
    """
    Base class for knowledge integration.
    
    Provides common functionality for integrating knowledge into different contexts.
    """
    
    def __init__(self):
        """Initialize a knowledge integrator."""
        pass
    
    def integrate(self, target: Any, knowledge: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> Any:
        """
        Integrate knowledge into a target.
        
        Args:
            target: The target to integrate knowledge into.
            knowledge: Knowledge item(s) to integrate.
            **kwargs: Additional integration parameters.
            
        Returns:
            Target with integrated knowledge.
        """
        raise NotImplementedError("Subclasses must implement integrate method")
    
    def format_knowledge(self, knowledge: Dict[str, Any], format_type: str = "default") -> str:
        """
        Format a knowledge item as text.
        
        Args:
            knowledge: Knowledge item to format.
            format_type: Type of formatting to apply.
            
        Returns:
            Formatted knowledge text.
        """
        k_type = knowledge.get("type", "")
        
        if format_type == "brief":
            # Brief format - just the core statement
            return knowledge.get("statement", "")
        
        elif format_type == "detailed":
            # Detailed format based on knowledge type
            if k_type == "conceptual_knowledge":
                return self._format_conceptual(knowledge, detailed=True)
            elif k_type == "procedural_knowledge":
                return self._format_procedural(knowledge, detailed=True)
            elif k_type == "format_specification":
                return self._format_format_spec(knowledge, detailed=True)
            elif k_type == "entity_classification":
                return self._format_entity_class(knowledge, detailed=True)
            elif k_type == "boundary_knowledge":
                return self._format_boundary(knowledge, detailed=True)
            else:
                return knowledge.get("statement", "")
        
        elif format_type == "contrastive":
            # Format with contrasting elements
            if k_type == "entity_classification":
                return self._format_entity_contrast(knowledge)
            elif k_type == "boundary_knowledge":
                return self._format_boundary_contrast(knowledge)
            else:
                return self._format_general_contrast(knowledge)
        
        elif format_type == "rule":
            # Format as a rule
            return self._format_as_rule(knowledge)
        
        elif format_type == "example":
            # Format as an example
            return self._format_as_example(knowledge)
        
        else:  # default format
            # Standard format based on knowledge type
            if k_type == "conceptual_knowledge":
                return self._format_conceptual(knowledge)
            elif k_type == "procedural_knowledge":
                return self._format_procedural(knowledge)
            elif k_type == "format_specification":
                return self._format_format_spec(knowledge)
            elif k_type == "entity_classification":
                return self._format_entity_class(knowledge)
            elif k_type == "boundary_knowledge":
                return self._format_boundary(knowledge)
            else:
                return knowledge.get("statement", "")
    
    def _format_conceptual(self, knowledge: Dict[str, Any], detailed: bool = False) -> str:
        """Format conceptual knowledge."""
        statement = knowledge.get("statement", "")
        entities = knowledge.get("entities", [])
        relations = knowledge.get("relations", [])
        
        if not detailed:
            return statement
        
        # Detailed format
        lines = [statement]
        
        if entities:
            lines.append(f"Key concepts: {', '.join(entities)}")
            
        if relations:
            rel_lines = []
            for rel in relations:
                subj = rel.get("subject", "")
                pred = rel.get("predicate", "")
                obj = rel.get("object", "")
                
                if subj and pred and obj:
                    # Format relation based on predicate type
                    if pred == "isA":
                        rel_lines.append(f"- {subj} is a type of {obj}")
                    elif pred == "isDefinedAs":
                        rel_lines.append(f"- {subj} is defined as {obj}")
                    elif pred == "hasProperty":
                        rel_lines.append(f"- {subj} has the property: {obj}")
                    elif pred == "differentiateFrom":
                        rel_lines.append(f"- {subj} should be differentiated from {obj}")
                    else:
                        rel_lines.append(f"- {subj} {pred} {obj}")
            
            if rel_lines:
                lines.append("Relationships:")
                lines.extend(rel_lines)
        
        return "\n".join(lines)
    
    def _format_procedural(self, knowledge: Dict[str, Any], detailed: bool = False) -> str:
        """Format procedural knowledge."""
        statement = knowledge.get("statement", "")
        topic = knowledge.get("procedure_topic", "")
        steps = knowledge.get("procedure_steps", [])
        
        if not detailed:
            if steps:
                step_text = "\n".join([f"- {step}" for step in steps[:3]])
                if len(steps) > 3:
                    step_text += f"\n- ... ({len(steps) - 3} more steps)"
                return f"{statement}\n{step_text}"
            else:
                return statement
        
        # Detailed format
        lines = [statement]
        
        if topic and topic not in statement:
            lines.append(f"Process: {topic}")
            
        if steps:
            lines.append("Steps:")
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
        
        return "\n".join(lines)
    
    def _format_format_spec(self, knowledge: Dict[str, Any], detailed: bool = False) -> str:
        """Format format specification knowledge."""
        statement = knowledge.get("statement", "")
        format_rules = knowledge.get("format_rules", [])
        
        if not detailed:
            return statement
        
        # Detailed format
        lines = [statement]
        
        if format_rules:
            lines.append("Format requirements:")
            for rule in format_rules:
                lines.append(f"- {rule}")
        
        return "\n".join(lines)
    
    def _format_entity_class(self, knowledge: Dict[str, Any], detailed: bool = False) -> str:
        """Format entity classification knowledge."""
        statement = knowledge.get("statement", "")
        entities = knowledge.get("entities", [])
        relations = knowledge.get("relations", [])
        
        if not detailed:
            return statement
        
        # Detailed format
        lines = [statement]
        
        if entities:
            lines.append(f"Entities: {', '.join(entities)}")
            
        if relations:
            for rel in relations:
                subj = rel.get("subject", "")
                pred = rel.get("predicate", "")
                obj = rel.get("object", "")
                
                if subj and pred == "differentiateFrom" and obj:
                    lines.append(f"Attention: Distinguish {subj} from {obj}")
        
        return "\n".join(lines)
    
    def _format_boundary(self, knowledge: Dict[str, Any], detailed: bool = False) -> str:
        """Format boundary knowledge."""
        statement = knowledge.get("statement", "")
        cases = knowledge.get("boundary_cases", [])
        
        if not detailed:
            return statement
        
        # Detailed format
        lines = [statement]
        
        if cases:
            lines.append("Special cases to consider:")
            for case in cases:
                lines.append(f"- {case}")
        
        return "\n".join(lines)
    
    def _format_entity_contrast(self, knowledge: Dict[str, Any]) -> str:
        """Format entity classification with contrast."""
        entities = knowledge.get("entities", [])
        relations = knowledge.get("relations", [])
        
        if not entities or not relations:
            return knowledge.get("statement", "")
        
        # Look for differentiation relations
        diff_rels = [rel for rel in relations if rel.get("predicate") == "differentiateFrom"]
        
        if diff_rels:
            lines = []
            
            for rel in diff_rels:
                subj = rel.get("subject", "")
                obj = rel.get("object", "")
                
                if subj and obj:
                    lines.append(f"Differentiate: {subj} vs. {obj}")
            
            return "\n".join(lines)
        
        return knowledge.get("statement", "")
    
    def _format_boundary_contrast(self, knowledge: Dict[str, Any]) -> str:
        """Format boundary knowledge with contrast."""
        cases = knowledge.get("boundary_cases", [])
        
        if not cases:
            return knowledge.get("statement", "")
        
        lines = ["Important distinctions:"]
        
        for case in cases:
            lines.append(f"- {case}")
        
        return "\n".join(lines)
    
    def _format_general_contrast(self, knowledge: Dict[str, Any]) -> str:
        """Format general knowledge with contrast."""
        statement = knowledge.get("statement", "")
        
        # Look for contrast indicators
        contrast_terms = ["not", "instead", "rather than", "as opposed to", "different from"]
        
        if any(term in statement.lower() for term in contrast_terms):
            return f"Note the distinction: {statement}"
        
        return statement
    
    def _format_as_rule(self, knowledge: Dict[str, Any]) -> str:
        """Format knowledge as a rule."""
        statement = knowledge.get("statement", "")
        k_type = knowledge.get("type", "")
        
        # Reformulate as a rule
        if "must" in statement or "should" in statement or "always" in statement or "never" in statement:
            # Already has rule-like language
            return f"Rule: {statement}"
        
        if k_type == "conceptual_knowledge":
            entities = knowledge.get("entities", [])
            if entities:
                return f"Rule: When dealing with {entities[0]}, {statement}"
            else:
                return f"Rule: {statement}"
        
        elif k_type == "entity_classification":
            entities = knowledge.get("entities", [])
            if entities:
                return f"Rule: Always correctly identify {entities[0]} as specified in: {statement}"
            else:
                return f"Rule: {statement}"
        
        elif k_type == "format_specification":
            return f"Rule: Always follow this format: {statement}"
        
        else:
            return f"Rule: {statement}"
    
    def _format_as_example(self, knowledge: Dict[str, Any]) -> str:
        """Format knowledge as an example."""
        k_type = knowledge.get("type", "")
        
        if k_type == "conceptual_knowledge":
            entities = knowledge.get("entities", [])
            statement = knowledge.get("statement", "")
            
            if entities:
                return f"Example: '{entities[0]}' - {statement}"
            else:
                return f"Example: {statement}"
        
        elif k_type == "procedural_knowledge":
            steps = knowledge.get("procedure_steps", [])
            topic = knowledge.get("procedure_topic", "")
            
            if steps and topic:
                step_text = "\n".join([f"  {i}. {step}" for i, step in enumerate(steps[:3], 1)])
                if len(steps) > 3:
                    step_text += f"\n  ... ({len(steps) - 3} more steps)"
                return f"Example process for {topic}:\n{step_text}"
            else:
                return f"Example: {knowledge.get('statement', '')}"
        
        elif k_type == "entity_classification":
            entities = knowledge.get("entities", [])
            statement = knowledge.get("statement", "")
            
            if entities:
                return f"Example distinction: {statement}"
            else:
                return f"Example: {statement}"
        
        else:
            return f"Example: {knowledge.get('statement', '')}"


class PromptKnowledgeIntegrator(KnowledgeIntegrator):
    """
    Integrate knowledge into prompts.
    
    This integrator specializes in adding knowledge to prompt states in an optimal way.
    """
    
    def __init__(
        self,
        placement_strategy: Optional[PlacementStrategy] = None,
        format_strategy: Optional[FormatSelectionStrategy] = None,
        conflict_strategy: Optional[ConflictResolutionStrategy] = None
    ):
        """
        Initialize a prompt knowledge integrator.
        
        Args:
            placement_strategy: Strategy for knowledge placement.
            format_strategy: Strategy for format selection.
            conflict_strategy: Strategy for conflict resolution.
        """
        super().__init__()
        self.placement_strategy = placement_strategy or PlacementStrategy()
        self.format_strategy = format_strategy or FormatSelectionStrategy()
        self.conflict_strategy = conflict_strategy or ConflictResolutionStrategy()
        
    def integrate(
        self, 
        prompt_state: PromptState, 
        knowledge: Union[Dict[str, Any], List[Dict[str, Any]]],
        **kwargs
    ) -> PromptState:
        """
        Integrate knowledge into a prompt state.
        
        Args:
            prompt_state: Prompt state to integrate knowledge into.
            knowledge: Knowledge item or list of items to integrate.
            **kwargs: Additional integration parameters.
                - max_items: Maximum number of knowledge items to integrate.
                - override_placement: Optional override of placement strategy.
                - override_format: Optional override of format strategy.
                
        Returns:
            Updated prompt state with integrated knowledge.
        """
        # Convert single knowledge item to list
        if not isinstance(knowledge, list):
            knowledge_items = [knowledge]
        else:
            knowledge_items = knowledge
        
        # Limit number of items if specified
        max_items = kwargs.get("max_items", len(knowledge_items))
        if max_items < len(knowledge_items):
            knowledge_items = knowledge_items[:max_items]
        
        # Get current prompt text
        prompt_text = prompt_state.text
        
        # Check for existing knowledge
        existing_knowledge = self._extract_existing_knowledge(prompt_text)
        
        # Resolve conflicts with existing knowledge
        knowledge_items = self.conflict_strategy.resolve_conflicts(
            knowledge_items, existing_knowledge)
        
        # If no knowledge items after conflict resolution, return original state
        if not knowledge_items:
            logger.debug("No knowledge items to integrate after conflict resolution")
            return prompt_state
        
        # Get override parameters
        override_placement = kwargs.get("override_placement")
        override_format = kwargs.get("override_format")
        
        # Process each knowledge item
        for item in knowledge_items:
            # Determine optimal placement
            if override_placement:
                placement = override_placement
            else:
                placement = self.placement_strategy.select_placement(item, prompt_state)
            
            # Determine optimal format
            if override_format:
                format_type = override_format
            else:
                format_type = self.format_strategy.select_format(item, prompt_state, placement)
            
            # Format knowledge text
            knowledge_text = self.format_knowledge(item, format_type)
            
            # Integrate knowledge at the determined placement
            prompt_text = self._integrate_at_placement(prompt_text, knowledge_text, placement)
        
        # Create new prompt state
        new_state = PromptState(
            text=prompt_text,
            history=prompt_state.history + ["knowledge_integration"],
            parent=prompt_state,
            action_applied="integrate_knowledge"
        )
        
        return new_state
    
    def _extract_existing_knowledge(self, prompt_text: str) -> List[str]:
        """Extract existing knowledge sections from prompt text."""
        # Look for knowledge sections
        knowledge_markers = [
            r"Domain Knowledge:(?:\s*\n)?(.*?)(?:\n\n|\Z)",
            r"Knowledge:(?:\s*\n)?(.*?)(?:\n\n|\Z)",
            r"Note:(?:\s*\n)?(.*?)(?:\n\n|\Z)",
            r"Important:(?:\s*\n)?(.*?)(?:\n\n|\Z)"
        ]
        
        existing = []
        
        for pattern in knowledge_markers:
            matches = re.findall(pattern, prompt_text, re.DOTALL)
            for match in matches:
                # Clean up and add to existing knowledge
                knowledge = match.strip()
                if knowledge:
                    existing.append(knowledge)
        
        return existing
    
    def _integrate_at_placement(self, prompt_text: str, knowledge_text: str, placement: str) -> str:
        """Integrate knowledge at the specified placement."""
        if placement == "knowledge_section":
            return self._add_to_knowledge_section(prompt_text, knowledge_text)
        
        elif placement == "role_description":
            return self._add_to_role(prompt_text, knowledge_text)
        
        elif placement == "task_description":
            return self._add_to_task(prompt_text, knowledge_text)
        
        elif placement == "step_instructions":
            return self._add_to_steps(prompt_text, knowledge_text)
        
        elif placement == "format_instructions":
            return self._add_to_format(prompt_text, knowledge_text)
        
        elif placement == "constraints":
            return self._add_to_constraints(prompt_text, knowledge_text)
        
        elif placement == "examples":
            return self._add_to_examples(prompt_text, knowledge_text)
        
        else:  # Default to adding at end
            return self._add_to_end(prompt_text, knowledge_text)
    
    def _add_to_knowledge_section(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to a dedicated knowledge section."""
        # Check if a knowledge section already exists
        knowledge_section_pattern = r"(Domain Knowledge:(?:\s*\n)?.*?)(\n\n|\Z)"
        match = re.search(knowledge_section_pattern, prompt_text, re.DOTALL)
        
        if match:
            # Add to existing section
            existing_section = match.group(1)
            
            # Check if the knowledge is already there (avoid duplication)
            if knowledge_text in existing_section:
                return prompt_text
                
            # Add to section
            updated_section = f"{existing_section}\n- {knowledge_text}"
            return prompt_text.replace(match.group(1), updated_section)
        
        # No existing section, create a new one
        # Try to insert after a logical section
        sections = ["Role:", "Task:", "Steps:", "Output Format:"]
        
        for section in reversed(sections):  # Start from end for better placement
            if section in prompt_text:
                parts = prompt_text.split(section, 1)
                section_content = parts[1].split("\n\n", 1)
                
                if len(section_content) > 1:
                    # Insert after this section
                    return f"{parts[0]}{section}{section_content[0]}\n\nDomain Knowledge:\n- {knowledge_text}\n\n{section_content[1]}"
        
        # If no suitable section found, add before the last paragraph
        if "\n\n" in prompt_text:
            parts = prompt_text.rsplit("\n\n", 1)
            return f"{parts[0]}\n\nDomain Knowledge:\n- {knowledge_text}\n\n{parts[1]}"
        
        # Last resort: add at the end
        return f"{prompt_text}\n\nDomain Knowledge:\n- {knowledge_text}"
    
    def _add_to_role(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to the role description."""
        role_pattern = r"(Role:(?:.*?)(?:\n\n|\Z))"
        match = re.search(role_pattern, prompt_text, re.DOTALL)
        
        if match:
            role_section = match.group(1)
            
            # Check if knowledge is already there
            if knowledge_text in role_section:
                return prompt_text
                
            # Add to role
            if role_section.strip().endswith("."):
                updated_role = f"{role_section[:-1]} with knowledge of {knowledge_text}."
            else:
                updated_role = f"{role_section.rstrip()} with knowledge of {knowledge_text}."
                
            return prompt_text.replace(match.group(1), updated_role)
        
        # No role section, add it
        if prompt_text.startswith("Task:") or prompt_text.startswith("Steps:"):
            return f"Role: Expert with knowledge of {knowledge_text}\n\n{prompt_text}"
        
        # Otherwise add at the beginning
        return f"Role: Expert with knowledge of {knowledge_text}\n\n{prompt_text}"
    
    def _add_to_task(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to the task description."""
        task_pattern = r"(Task:(?:.*?)(?:\n\n|\Z))"
        match = re.search(task_pattern, prompt_text, re.DOTALL)
        
        if match:
            task_section = match.group(1)
            
            # Check if knowledge is already there
            if knowledge_text in task_section:
                return prompt_text
                
            # Add to task
            if task_section.strip().endswith("."):
                updated_task = f"{task_section[:-1]}. Note: {knowledge_text}."
            else:
                updated_task = f"{task_section.rstrip()}. Note: {knowledge_text}."
                
            return prompt_text.replace(match.group(1), updated_task)
        
        # No task section found
        return prompt_text
    
    def _add_to_steps(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to step instructions."""
        steps_pattern = r"(Steps:(?:[\s\S]*?)(?:\n\n|\Z))"
        match = re.search(steps_pattern, prompt_text, re.DOTALL)
        
        if match:
            steps_section = match.group(1)
            
            # Check if knowledge is already there
            if knowledge_text in steps_section:
                return prompt_text
                
            # Add as a new step or note
            if re.search(r"\d+\.", steps_section) or re.search(r"- ", steps_section):
                # Numbered or bulleted steps
                if steps_section.strip().endswith("."):
                    updated_steps = f"{steps_section}\n- Important: {knowledge_text}"
                else:
                    updated_steps = f"{steps_section}\n- Important: {knowledge_text}"
            else:
                # No visible step structure
                if steps_section.strip().endswith("."):
                    updated_steps = f"{steps_section} Important: {knowledge_text}."
                else:
                    updated_steps = f"{steps_section} Important: {knowledge_text}."
                
            return prompt_text.replace(match.group(1), updated_steps)
        
        # No steps section found
        return prompt_text
    
    def _add_to_format(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to format instructions."""
        format_pattern = r"((?:Output )?Format:(?:[\s\S]*?)(?:\n\n|\Z))"
        match = re.search(format_pattern, prompt_text, re.DOTALL)
        
        if match:
            format_section = match.group(1)
            
            # Check if knowledge is already there
            if knowledge_text in format_section:
                return prompt_text
                
            # Add to format instructions
            if format_section.strip().endswith("."):
                updated_format = f"{format_section} {knowledge_text}."
            else:
                updated_format = f"{format_section}. {knowledge_text}."
                
            return prompt_text.replace(match.group(1), updated_format)
        
        # No format section found
        return prompt_text
    
    def _add_to_constraints(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to constraints section."""
        constraints_pattern = r"(Constraints:(?:[\s\S]*?)(?:\n\n|\Z))"
        match = re.search(constraints_pattern, prompt_text, re.DOTALL)
        
        if match:
            constraints_section = match.group(1)
            
            # Check if knowledge is already there
            if knowledge_text in constraints_section:
                return prompt_text
                
            # Add as a new constraint
            if re.search(r"\d+\.", constraints_section) or re.search(r"- ", constraints_section):
                # Numbered or bulleted constraints
                updated_constraints = f"{constraints_section}\n- {knowledge_text}"
            else:
                # No visible list structure
                if constraints_section.strip().endswith("."):
                    updated_constraints = f"{constraints_section} {knowledge_text}."
                else:
                    updated_constraints = f"{constraints_section}. {knowledge_text}."
                
            return prompt_text.replace(match.group(1), updated_constraints)
        
        # No constraints section, add new one
        # Try to insert after a logical section
        sections = ["Role:", "Task:", "Steps:", "Output Format:"]
        
        for section in reversed(sections):  # Start from end for better placement
            if section in prompt_text:
                parts = prompt_text.split(section, 1)
                section_content = parts[1].split("\n\n", 1)
                
                if len(section_content) > 1:
                    # Insert after this section
                    return f"{parts[0]}{section}{section_content[0]}\n\nConstraints:\n- {knowledge_text}\n\n{section_content[1]}"
        
        # Last resort: add at the end
        return f"{prompt_text}\n\nConstraints:\n- {knowledge_text}"
    
    def _add_to_examples(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to examples section."""
        examples_pattern = r"(Examples?:(?:[\s\S]*?)(?:\n\n|\Z))"
        match = re.search(examples_pattern, prompt_text, re.DOTALL)
        
        if match:
            examples_section = match.group(1)
            
            # Check if knowledge is already there
            if knowledge_text in examples_section:
                return prompt_text
                
            # Add as a new example or note
            updated_examples = f"{examples_section}\n\nAdditional Example:\n{knowledge_text}"
            return prompt_text.replace(match.group(1), updated_examples)
        
        # No examples section, add new one at a logical position
        if "Output Format:" in prompt_text:
            parts = prompt_text.split("Output Format:", 1)
            format_content = parts[1].split("\n\n", 1)
            
            if len(format_content) > 1:
                # Insert after output format
                return f"{parts[0]}Output Format:{format_content[0]}\n\nExample:\n{knowledge_text}\n\n{format_content[1]}"
            else:
                # Add after output format at the end
                return f"{prompt_text}\n\nExample:\n{knowledge_text}"
        
        # Otherwise add at the end
        return f"{prompt_text}\n\nExample:\n{knowledge_text}"
    
    def _add_to_end(self, prompt_text: str, knowledge_text: str) -> str:
        """Add knowledge to the end of the prompt."""
        return f"{prompt_text}\n\nImportant: {knowledge_text}"