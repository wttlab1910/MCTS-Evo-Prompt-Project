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
            "boundary_confusion": ["add_constraint", "add_example", "add_domain_knowledge"],
            "calculation_error": ["modify_workflow", "add_explanation", "add_example"],
            "missing_context": ["add_domain_knowledge", "add_constraint", "add_explanation"],
            "lookup_error": ["modify_workflow", "add_example", "add_explanation"],
            "classification_error": ["add_domain_knowledge", "add_constraint", "clarify_terminology"],
            "numerical_error": ["add_explanation", "add_constraint", "add_example"]
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
            ],
            "calculation_error": [
                "Provide a step-by-step calculation approach for {issue}",
                "Add verification steps for calculations involving {issue}",
                "Include example calculations similar to {issue}"
            ],
            "missing_context": [
                "Add context information about {issue}",
                "Explain the importance of considering all aspects of {issue}",
                "Include examples that demonstrate proper context handling for {issue}"
            ],
            "lookup_error": [
                "Improve table/data lookup instructions for {issue}",
                "Add step-by-step lookup procedure for {issue}",
                "Include example of proper data extraction for {issue}"
            ],
            "classification_error": [
                "Clarify classification criteria for {issue}",
                "Add domain knowledge about categories related to {issue}",
                "Include examples of proper classification for {issue}"
            ],
            "numerical_error": [
                "Add precision requirements for calculations involving {issue}",
                "Include verification steps for numerical operations on {issue}",
                "Provide calculation examples for {issue}"
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
            # If no patterns found in analysis, try to extract from raw errors
            raw_errors = analysis.get("errors", [])
            if raw_errors:
                patterns = self._generate_basic_patterns(raw_errors)
            
            logger.debug(f"No patterns found in analysis, generated {len(patterns)} basic patterns from {len(raw_errors)} raw errors")
            
            if not patterns:
                logger.debug("No patterns or raw errors found in analysis")
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
    
    def _generate_basic_patterns(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate basic patterns from raw errors when no patterns were found.
        
        Args:
            errors: List of raw error dictionaries.
            
        Returns:
            List of generated pattern dictionaries.
        """
        if not errors:
            return []
            
        # Group errors by type
        error_types = {}
        for error in errors:
            error_type = error.get("error_type", "unknown")
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        # Create a pattern for each error type
        basic_patterns = []
        for error_type, type_errors in error_types.items():
            if not type_errors:
                continue
                
            # Extract entities from errors
            entities = set()
            for error in type_errors:
                for entity in self._extract_entities_from_error(error):
                    entities.add(entity)
            
            # Create pattern
            pattern = {
                "pattern_type": error_type,
                "description": self._generate_description_from_errors(type_errors),
                "entities": list(entities),
                "frequency": len(type_errors)
            }
            basic_patterns.append(pattern)
        
        return basic_patterns
    
    def _extract_entities_from_error(self, error: Dict[str, Any]) -> List[str]:
        """
        Extract potential entities from an error.
        
        Args:
            error: Error dictionary.
            
        Returns:
            List of extracted entities.
        """
        entities = []
        
        # Extract from example
        example = error.get("example", {})
        if isinstance(example, dict):
            example_text = example.get("text", "")
        else:
            example_text = str(example)
            
        # Extract from actual vs expected
        expected = error.get("expected", "")
        actual = error.get("actual", "")
        
        # Look for potential entities
        for text in [example_text, expected, actual]:
            if isinstance(text, str):
                # Look for capitalized words, numbers, and words in quotes
                cap_matches = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\b', text)
                quote_matches = re.findall(r'"([^"]+)"', text)
                number_matches = re.findall(r'\b(\d+)\b', text)
                
                entities.extend(cap_matches)
                entities.extend(quote_matches)
                entities.extend(number_matches)
        
        # Remove duplicates and return
        return list(set(entities))
    
    def _generate_description_from_errors(self, errors: List[Dict[str, Any]]) -> str:
        """
        Generate a descriptive summary from a list of errors.
        
        Args:
            errors: List of error dictionaries.
            
        Returns:
            Error description string.
        """
        if not errors:
            return "Unknown error pattern"
            
        # Try to get existing description
        descriptions = [e.get("description", "") for e in errors if e.get("description")]
        if descriptions:
            return max(descriptions, key=len)  # Return the longest description
        
        # Generate based on error type and content
        error_type = errors[0].get("error_type", "unknown")
        
        # Get expected vs actual values from the first error
        first_error = errors[0]
        expected = first_error.get("expected", "")
        actual = first_error.get("actual", "")
        
        if error_type == "entity_confusion":
            return f"Confusion between entity types or categories. Expected: {expected}, Got: {actual}"
        elif error_type == "calculation_error":
            return f"Error in numerical calculations or counting. Expected: {expected}, Got: {actual}"
        elif error_type == "format_error" or error_type == "format_inconsistency":
            return f"Output format does not match requirements. Expected: {expected}, Got: {actual}"
        elif error_type == "missing_context":
            return "Missing important contextual information needed to answer correctly."
        elif error_type == "classification_error":
            return f"Incorrect classification of items. Expected: {expected}, Got: {actual}"
        elif error_type == "lookup_error":
            return "Incorrect information extraction or lookup from data."
        elif error_type == "numerical_error":
            return f"Incorrect numerical value. Expected: {expected}, Got: {actual}"
        else:
            return f"Error pattern of type {error_type}. Expected: {expected}, Got: {actual}"
    
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
        action_mapping = self._map_to_action(action_type, description, entities, issue, pattern_type)
        
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
            
        # Higher impact for calculation errors
        if pattern_type == "calculation_error":
            return "High - Will improve numerical accuracy"
            
        # Higher impact for format inconsistencies
        if pattern_type == "format_inconsistency":
            return "Medium - Will standardize output format"
            
        # Higher impact for classification errors
        if pattern_type == "classification_error":
            return "High - Will improve classification accuracy"
            
        # Higher impact for lookup errors in table tasks
        if pattern_type == "lookup_error":
            return "High - Will improve data extraction accuracy"
            
        # General impact
        return "Medium - Will improve accuracy and clarity"
    
    def _map_to_action(self, action_type: str, description: str, entities: List[str], issue: str, pattern_type: str) -> Dict[str, Any]:
        """
        Map feedback to a specific action.
        
        Args:
            action_type: Type of action.
            description: Error description.
            entities: Entities involved.
            issue: Key issue identified.
            pattern_type: Type of error pattern.
            
        Returns:
            Action mapping dictionary.
        """
        action_mapping = {
            "action_type": action_type,
            "parameters": {}
        }
        
        # Enhance action mapping based on error pattern type
        if pattern_type == "entity_confusion":
            if action_type == "add_domain_knowledge":
                action_mapping["parameters"] = {
                    "knowledge_text": f"Clarify the distinction between different types: {issue}",
                    "domain": "entity_classification"
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
                    "constraint_text": f"Pay careful attention to entity types when identifying {issue}"
                }
        
        elif pattern_type == "calculation_error":
            if action_type == "modify_workflow":
                action_mapping["parameters"] = {
                    "steps": [
                        f"Clearly identify all numeric values needed for {issue}",
                        f"Perform calculations step-by-step, showing your work",
                        f"Verify calculations by double-checking for {issue}"
                    ]
                }
            elif action_type == "add_explanation":
                action_mapping["parameters"] = {
                    "explanation_text": f"Always verify numerical calculations for {issue} by showing your work step-by-step",
                    "target": "task"
                }
            elif action_type == "add_example":
                action_mapping["parameters"] = {
                    "example_text": f"Example calculation for {issue}: Show all steps of the calculation process, including intermediate values.",
                    "example_type": "calculation"
                }
        
        elif pattern_type == "lookup_error":
            if action_type == "modify_workflow":
                action_mapping["parameters"] = {
                    "steps": [
                        f"Carefully identify the table columns and structure",
                        f"Look for the specific information related to {issue}",
                        f"Extract the exact value(s) needed",
                        f"Verify that the extracted information is correct"
                    ]
                }
            elif action_type == "add_explanation":
                action_mapping["parameters"] = {
                    "explanation_text": f"When extracting information from tables, carefully identify the correct row and column for {issue}",
                    "target": "task" 
                }
        
        elif pattern_type == "classification_error":
            if action_type == "add_domain_knowledge":
                action_mapping["parameters"] = {
                    "knowledge_text": f"When classifying {issue}, carefully consider the defining characteristics of each category",
                    "domain": "classification"
                }
            elif action_type == "add_constraint":
                action_mapping["parameters"] = {
                    "constraint_text": f"Verify classification decisions for {issue} against the category definitions"
                }
        
        # Set parameters for standard action types if not already set
        if not action_mapping["parameters"]:
            if action_type == "add_domain_knowledge":
                action_mapping["parameters"] = {
                    "knowledge_text": description,
                    "domain": "general"
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
        
        # Extract from Expected vs Actual pattern
        expected_actual_match = re.search(r'Expected: (.*?), Got: (.*?)$', description)
        if expected_actual_match:
            expected = expected_actual_match.group(1).strip()
            return f"{term} refers to {expected}, not {expected_actual_match.group(2).strip()}"
        
        # Fall back to using the description
        return f"{term} is important in this context: {description}"
    
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
        if "Expected: " in description and "Got: " in description:
            # Extract from description
            match = re.search(r'Expected: (.*?), Got: (.*?)$', description)
            if match:
                expected = match.group(1).strip()
                actual = match.group(2).strip()
                return f"Example: When given this problem, respond with '{expected}' not '{actual}'."
        
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