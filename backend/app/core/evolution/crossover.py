"""
Crossover operations for evolutionary prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import random
import re
from app.core.mdp.state import PromptState
from app.utils.logger import get_logger

logger = get_logger("evolution.crossover")

class PromptCrossover:
    """
    Implements crossover operations for prompt states.
    
    Crossovers combine elements from two parent prompts to create
    a new prompt that potentially inherits beneficial traits from both.
    """
    
    def __init__(self):
        """Initialize a prompt crossover operator."""
        logger.debug("Initialized PromptCrossover")
    
    def crossover(self, parent1: PromptState, parent2: PromptState) -> PromptState:
        """
        Perform crossover between two parent prompt states.
        
        Args:
            parent1: First parent prompt state.
            parent2: Second parent prompt state.
            
        Returns:
            New prompt state created by combining elements from both parents.
        """
        # Select a crossover operator based on the parents' characteristics
        crossover_operators = [
            self._component_crossover,
            self._section_crossover,
            self._alternating_lines_crossover
        ]
        
        # Choose an operator
        operator = random.choice(crossover_operators)
        
        # Apply the selected crossover
        crossed_text = operator(parent1, parent2)
        
        # Create a new state with the crossed text
        history = []
        if parent1.history:
            history.extend(parent1.history)
        history.append(f"crossover with {parent2.state_id[:8]}")
        
        crossed_state = PromptState(
            text=crossed_text,
            history=history,
            parent=parent1,
            action_applied="crossover"
        )
        
        logger.debug(f"Applied {operator.__name__} crossover between states "
                    f"{parent1.state_id[:8]} and {parent2.state_id[:8]}")
        return crossed_state
    
    def _component_crossover(self, parent1: PromptState, parent2: PromptState) -> str:
        """
        Combine components from both parents.
        
        Args:
            parent1: First parent prompt state.
            parent2: Second parent prompt state.
            
        Returns:
            Text created by combining components from both parents.
        """
        # Get components from both parents
        components1 = parent1.components
        components2 = parent2.components
        
        # Create a combined component dictionary
        combined = {}
        
        # For each component type, choose which parent to inherit from
        for component in set(list(components1.keys()) + list(components2.keys())):
            has_in_1 = component in components1 and components1[component]
            has_in_2 = component in components2 and components2[component]
            
            if has_in_1 and has_in_2:
                # If both parents have the component, randomly choose
                if random.random() < 0.5:
                    combined[component] = components1[component]
                else:
                    combined[component] = components2[component]
            elif has_in_1:
                # Only parent1 has it
                combined[component] = components1[component]
            elif has_in_2:
                # Only parent2 has it
                combined[component] = components2[component]
        
        # Reconstruct text from components
        # This is simplified and would need more sophistication in a real implementation
        result = []
        
        # Add role if present
        if "role" in combined:
            result.append(f"Role: {combined['role']}")
        
        # Add task if present
        if "task" in combined:
            result.append(f"Task: {combined['task']}")
        
        # Add steps if present
        if "steps" in combined and combined["steps"]:
            steps_text = "Steps:"
            for step in combined["steps"]:
                steps_text += f"\n- {step}"
            result.append(steps_text)
        
        # Add output_format if present
        if "output_format" in combined:
            result.append(f"Output Format: {combined['output_format']}")
        
        # Add examples if present
        if "examples" in combined and combined["examples"]:
            if isinstance(combined["examples"], list):
                if len(combined["examples"]) == 1:
                    result.append(f"Example:\n{combined['examples'][0]}")
                else:
                    examples_text = "Examples:"
                    for example in combined["examples"]:
                        if isinstance(example, dict) and "text" in example:
                            examples_text += f"\n\n{example['text']}"
                        else:
                            examples_text += f"\n\n{example}"
                    result.append(examples_text)
        
        # Add constraints if present
        if "constraints" in combined and combined["constraints"]:
            if isinstance(combined["constraints"], list):
                constraints_text = "Constraints:"
                for constraint in combined["constraints"]:
                    constraints_text += f"\n- {constraint}"
                result.append(constraints_text)
            else:
                result.append(f"Constraints: {combined['constraints']}")
        
        # Add domain_knowledge if present
        if "domain_knowledge" in combined and combined["domain_knowledge"]:
            if isinstance(combined["domain_knowledge"], list):
                knowledge_text = "Domain Knowledge:"
                for knowledge in combined["domain_knowledge"]:
                    if isinstance(knowledge, dict) and "text" in knowledge:
                        knowledge_text += f"\n- {knowledge['text']}"
                    else:
                        knowledge_text += f"\n- {knowledge}"
                result.append(knowledge_text)
            else:
                result.append(f"Domain Knowledge: {combined['domain_knowledge']}")
        
        return "\n\n".join(result)
    
    def _section_crossover(self, parent1: PromptState, parent2: PromptState) -> str:
        """
        Combine sections from both parents.
        
        Args:
            parent1: First parent prompt state.
            parent2: Second parent prompt state.
            
        Returns:
            Text created by combining sections from both parents.
        """
        # Split both parents into sections
        sections1 = parent1.text.split("\n\n")
        sections2 = parent2.text.split("\n\n")
        
        # Identify section types (role, task, steps, etc.)
        section_types1 = self._identify_section_types(sections1)
        section_types2 = self._identify_section_types(sections2)
        
        # Create a merged set of section types
        all_types = set(section_types1.keys()) | set(section_types2.keys())
        
        # For each section type, choose which parent to inherit from
        result = []
        for section_type in all_types:
            if section_type in section_types1 and section_type in section_types2:
                # Both parents have this section type, randomly choose
                if random.random() < 0.5:
                    result.append(sections1[section_types1[section_type]])
                else:
                    result.append(sections2[section_types2[section_type]])
            elif section_type in section_types1:
                # Only parent1 has it
                result.append(sections1[section_types1[section_type]])
            else:
                # Only parent2 has it
                result.append(sections2[section_types2[section_type]])
        
        return "\n\n".join(result)
    
    def _alternating_lines_crossover(self, parent1: PromptState, parent2: PromptState) -> str:
        """
        Create a new prompt by alternating lines from both parents.
        
        Args:
            parent1: First parent prompt state.
            parent2: Second parent prompt state.
            
        Returns:
            Text created by alternating lines from both parents.
        """
        # Split both parents into lines
        lines1 = parent1.text.split("\n")
        lines2 = parent2.text.split("\n")
        
        # Take lines from each parent, alternating between them
        # But keep consecutive lines that form a logical unit together
        
        result = []
        parent1_turn = True
        i1, i2 = 0, 0
        
        while i1 < len(lines1) or i2 < len(lines2):
            if parent1_turn and i1 < len(lines1):
                # Take lines from parent1
                line = lines1[i1].strip()
                result.append(line)
                i1 += 1
                
                # If this is a header line, take its content too
                if line and line.endswith(":") and i1 < len(lines1):
                    # Capture indented content that follows
                    while i1 < len(lines1) and (not lines1[i1].strip() or lines1[i1].startswith(" ") or lines1[i1].startswith("-")):
                        result.append(lines1[i1])
                        i1 += 1
            elif not parent1_turn and i2 < len(lines2):
                # Take lines from parent2
                line = lines2[i2].strip()
                result.append(line)
                i2 += 1
                
                # If this is a header line, take its content too
                if line and line.endswith(":") and i2 < len(lines2):
                    # Capture indented content that follows
                    while i2 < len(lines2) and (not lines2[i2].strip() or lines2[i2].startswith(" ") or lines2[i2].startswith("-")):
                        result.append(lines2[i2])
                        i2 += 1
            
            # Switch turns only if the current parent has lines left
            if (parent1_turn and i1 < len(lines1)) or (not parent1_turn and i2 < len(lines2)):
                parent1_turn = not parent1_turn
            
            # If one parent runs out of lines, just keep using the other
            if parent1_turn and i1 >= len(lines1):
                parent1_turn = False
            if not parent1_turn and i2 >= len(lines2):
                parent1_turn = True
        
        return "\n".join(result)
    
    def _identify_section_types(self, sections: List[str]) -> Dict[str, int]:
        """
        Identify the types of sections in a list of sections.
        
        Args:
            sections: List of text sections.
            
        Returns:
            Dictionary mapping section types to their indices.
        """
        section_types = {}
        
        for i, section in enumerate(sections):
            section_lower = section.lower()
            if section_lower.startswith("role:"):
                section_types["role"] = i
            elif section_lower.startswith("task:"):
                section_types["task"] = i
            elif section_lower.startswith("steps:"):
                section_types["steps"] = i
            elif section_lower.startswith("output format:"):
                section_types["output_format"] = i
            elif section_lower.startswith("example:"):
                section_types["example"] = i
            elif section_lower.startswith("examples:"):
                section_types["examples"] = i
            elif section_lower.startswith("constraints:"):
                section_types["constraints"] = i
            elif section_lower.startswith("domain knowledge:"):
                section_types["domain_knowledge"] = i
            elif section_lower.startswith("note:"):
                section_types["note"] = i
            
        return section_types