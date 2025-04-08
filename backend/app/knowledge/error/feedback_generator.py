"""
Error feedback generator.

This module generates actionable feedback from error analysis.
"""
from typing import Dict, Any, List, Optional, Set
import re
import random

from app.utils.logger import get_logger
from app.core.mdp.action import create_action

logger = get_logger("knowledge.error.feedback_generator")

class FeedbackGenerator:
    """
    Generate actionable feedback from error analysis.
    
    This generator creates feedback suggestions and maps them to specific actions.
    """
    
    def __init__(self):
        """Initialize a feedback generator."""
        # Map error patterns to action types
        self.error_action_mapping = {
            "entity_confusion": ["add_domain_knowledge", "clarify_terminology", "add_constraint"],
            "procedure_error": ["modify_workflow", "add_explanation", "add_example"],
            "domain_misconception": ["add_domain_knowledge", "add_explanation", "add_rule"],
            "format_inconsistency": ["specify_format", "add_template", "structure_output"],
            "boundary_confusion": ["add_constraint", "add_example", "add_domain_knowledge"]
        }
        
        # Suggestion templates for different error types
        self.suggestion_templates = {
            "entity_confusion": [
                "Clarify the distinction between {entities}",
                "Provide explicit definition for {entities}",
                "Add domain knowledge about {entities}"
            ],
            "procedure_error": [
                "Restructure the workflow to emphasize {issue}",
                "Add explicit instructions for {issue}",
                "Include examples showing correct {issue}"
            ],
            "domain_misconception": [
                "Add domain knowledge about {issue}",
                "Explain the concept of {issue} more clearly",
                "Include rules about {issue}"
            ],
            "format_inconsistency": [
                "Specify exact format requirements for {issue}",
                "Provide a template demonstrating {issue}",
                "Structure the output to clarify {issue}"
            ],
            "boundary_confusion": [
                "Add constraints to handle {issue}",
                "Include examples of edge cases like {issue}",
                "Provide domain knowledge about special cases like {issue}"
            ]
        }
        
    def generate_feedback(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate feedback from error analysis.
        
        Args:
            analysis: Error analysis dictionary.
            
        Returns:
            List of feedback items.
        """
        if not analysis:
            logger.warning("No analysis provided for feedback generation")
            return []
            
        patterns = analysis.get("patterns", [])
        if not patterns:
            logger.debug("No patterns found in analysis")
            return []
            
        feedback_items = []
        
        # Process each pattern
        for pattern in patterns:
            pattern_type = pattern.get("pattern_type", "")
            description = pattern.get("description", "")
            entities = pattern.get("entities", [])
            
            # Skip patterns with insufficient information
            if not pattern_type or not description:
                continue
                
            # Generate feedback for this pattern
            feedback = self._generate_pattern_feedback(pattern_type, description, entities)
            if feedback:
                feedback_items.append(feedback)
        
        logger.debug(f"Generated {len(feedback_items)} feedback items")
        return feedback_items
    
    def _generate_pattern_feedback(self, pattern_type: str, description: str, entities: List[str]) -> Optional[Dict[str, Any]]:
        """
        Generate feedback for a specific error pattern.
        
        Args:
            pattern_type: Type of error pattern.
            description: Error description.
            entities: Entities involved in the error.
            
        Returns:
            Feedback item or None if no suitable feedback.
        """
        # Determine the issue for suggestion template
        if entities:
            issue = ", ".join(entities)
        else:
            # Extract key terms from description
            noun_phrases = self._extract_noun_phrases(description)
            issue = ", ".join(noun_phrases[:2]) if noun_phrases else "this issue"
        
        # Select appropriate action type based on pattern type
        available_actions = self.error_action_mapping.get(pattern_type, [])
        if not available_actions:
            available_actions = ["add_explanation"]
            
        action_type = random.choice(available_actions)
        
        # Select suggestion template
        templates = self.suggestion_templates.get(pattern_type, ["Address {issue} more clearly"])
        suggestion_template = random.choice(templates)
        suggestion = suggestion_template.format(entities=issue, issue=issue)
        
        # Determine potential impact
        impact = self._estimate_impact(pattern_type, entities)
        
        # Generate action mapping
        action_mapping = self._map_to_action(action_type, description, entities, issue)
        
        # Create feedback item
        feedback = {
            "type": pattern_type,
            "description": description,
            "suggestion": suggestion,
            "impact": impact,
            "action_mapping": action_mapping
        }
        
        return feedback
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract potential noun phrases from text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            List of noun phrases.
        """
        # Simple noun phrase extraction
        # Look for adjective + noun or noun sequences
        matches = re.findall(r'\b([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)?)\b', text)
        
        # Also find single nouns that might be important
        single_nouns = re.findall(r'\b([A-Za-z]{4,})\b', text)
        
        # Combine and remove duplicates
        all_phrases = matches + single_nouns
        unique_phrases = []
        
        for phrase in all_phrases:
            if phrase.lower() not in [p.lower() for p in unique_phrases]:
                unique_phrases.append(phrase)
        
        return unique_phrases
    
    def _estimate_impact(self, pattern_type: str, entities: List[str]) -> str:
        """
        Estimate potential impact of addressing the issue.
        
        Args:
            pattern_type: Type of error pattern.
            entities: Entities involved.
            
        Returns:
            Impact description.
        """
        # Higher impact for entity confusion with multiple entities
        if pattern_type == "entity_confusion" and len(entities) > 1:
            return "High - Will improve entity recognition accuracy"
            
        # Higher impact for procedural errors
        if pattern_type == "procedure_error":
            return "High - Will improve process adherence"
            
        # Higher impact for format inconsistencies
        if pattern_type == "format_inconsistency":
            return "Medium - Will standardize output format"
            
        # General impact
        return "Medium - Will improve accuracy and clarity"
    
    def _map_to_action(self, action_type: str, description: str, entities: List[str], issue: str) -> Dict[str, Any]:
        """
        Map feedback to a specific action.
        
        Args:
            action_type: Type of action.
            description: Error description.
            entities: Entities involved.
            issue: Key issue identified.
            
        Returns:
            Action mapping dictionary.
        """
        action_mapping = {
            "action_type": action_type,
            "parameters": {}
        }
        
        # Set parameters based on action type
        if action_type == "add_domain_knowledge":
            action_mapping["parameters"] = {
                "knowledge_text": description,
                "domain": "general",
                "location": "knowledge_section"
            }
            
        elif action_type == "clarify_terminology":
            if entities:
                entity = entities[0]
                definition = self._extract_definition(entity, description)
                action_mapping["parameters"] = {
                    "term": entity,
                    "definition": definition
                }
            
        elif action_type == "add_constraint":
            action_mapping["parameters"] = {
                "constraint_text": description
            }
            
        elif action_type == "modify_workflow":
            steps = self._extract_steps(description)
            action_mapping["parameters"] = {
                "steps": steps
            }
            
        elif action_type == "add_explanation":
            action_mapping["parameters"] = {
                "explanation_text": description,
                "target": "task"
            }
            
        elif action_type == "add_example":
            example = self._generate_example(description, entities)
            action_mapping["parameters"] = {
                "example_text": example,
                "example_type": "input_output"
            }
            
        elif action_type == "add_rule":
            action_mapping["parameters"] = {
                "rule_text": f"Rule: {description}"
            }
            
        elif action_type == "specify_format":
            action_mapping["parameters"] = {
                "format_text": description
            }
            
        elif action_type == "add_template":
            action_mapping["parameters"] = {
                "template_text": f"Template for {issue}: {description}"
            }
            
        elif action_type == "structure_output":
            structure = self._suggest_structure(description)
            action_mapping["parameters"] = {
                "structure_text": structure
            }
        
        return action_mapping
    
    def _extract_definition(self, term: str, description: str) -> str:
        """
        Extract a definition for a term from description.
        
        Args:
            term: Term to define.
            description: Description text.
            
        Returns:
            Definition text.
        """
        # Look for patterns like "X is a Y" or "X should be Y"
        patterns = [
            rf"{term} is a(?:n)? (.+)",
            rf"{term} (?:refers to|means) (.+)",
            rf"{term} should be (.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fall back to using the description
        return description
    
    def _extract_steps(self, description: str) -> List[str]:
        """
        Extract potential steps from description.
        
        Args:
            description: Description text.
            
        Returns:
            List of steps.
        """
        # Split by sentence and convert to steps
        sentences = re.split(r'[.!?]\s', description)
        steps = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short sentences
                # Remove phrases like "you should" or "model should"
                step = re.sub(r'(?:you|model|it|one)\s+should\s+', '', sentence, flags=re.IGNORECASE)
                steps.append(step)
        
        if not steps:
            steps = ["Process this carefully"]
            
        return steps
    
    def _generate_example(self, description: str, entities: List[str]) -> str:
        """
        Generate an example based on description and entities.
        
        Args:
            description: Description text.
            entities: Entities involved.
            
        Returns:
            Example text.
        """
        if not entities:
            return f"Example: {description}"
            
        # Create a simple example using the entities
        entity_text = ", ".join(entities)
        
        return f"Input: Text containing {entity_text}\nOutput: Proper handling of {entity_text}"
    
    def _suggest_structure(self, description: str) -> str:
        """
        Suggest an output structure based on description.
        
        Args:
            description: Description text.
            
        Returns:
            Structure suggestion.
        """
        # Create a simple structure suggestion
        categories = re.findall(r'([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})', description)
        
        if categories:
            structure = "Structure the output as follows:\n"
            for category in categories[:3]:  # Limit to first three matches
                structure += f"- {category.strip().title()}: [relevant information]\n"
            return structure
        
        return "Structure the output into clear sections with headers"
    
    def map_feedback_to_actions(self, feedback_items: List[Dict[str, Any]]) -> List[Any]:
        """
        Map feedback items to concrete actions.
        
        Args:
            feedback_items: List of feedback items.
            
        Returns:
            List of actions.
        """
        actions = []
        
        for feedback in feedback_items:
            mapping = feedback.get("action_mapping", {})
            action_type = mapping.get("action_type")
            parameters = mapping.get("parameters", {})
            
            if action_type:
                try:
                    action = create_action(action_type, parameters=parameters)
                    actions.append(action)
                except Exception as e:
                    logger.error(f"Error creating action from feedback: {e}")
        
        logger.debug(f"Mapped {len(actions)} actions from {len(feedback_items)} feedback items")
        return actions