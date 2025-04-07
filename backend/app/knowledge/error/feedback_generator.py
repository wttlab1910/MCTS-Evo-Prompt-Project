"""
Feedback generation for prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple
from app.core.mdp.action import Action, create_action
from app.utils.logger import get_logger

logger = get_logger("error.feedback")

class FeedbackGenerator:
    """
    Generates improvement suggestions based on error analysis.
    
    This class handles creating specific improvement suggestions
    and mapping them to concrete actions that can be applied to prompts.
    """
    
    def __init__(self):
        """Initialize a feedback generator."""
        # Mapping from error categories to improvement strategies
        self.improvement_strategies = {
            "semantic_error": self._generate_semantic_feedback,
            "format_error": self._generate_format_feedback,
            "reasoning_error": self._generate_reasoning_feedback,
            "omission_error": self._generate_omission_feedback,
            "hallucination_error": self._generate_hallucination_feedback,
            "boundary_error": self._generate_boundary_feedback,
            "context_error": self._generate_context_feedback,
            "domain_error": self._generate_domain_feedback
        }
        
        logger.debug("Initialized FeedbackGenerator")
    
    def generate_feedback(
        self, 
        analysis: Dict[str, Any],
        max_suggestions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback based on error analysis.
        
        Args:
            analysis: Error analysis dictionary.
            max_suggestions: Maximum number of suggestions to generate.
            
        Returns:
            List of feedback dictionaries with improvement suggestions.
        """
        if not analysis or "error_clusters" not in analysis or not analysis["error_clusters"]:
            logger.warning("No error analysis to generate feedback from")
            return []
        
        feedback_items = []
        
        # Process each error cluster
        for category, errors in analysis["error_clusters"].items():
            if category in self.improvement_strategies:
                # Generate feedback for this category
                category_feedback = self.improvement_strategies[category](errors, analysis)
                if category_feedback:
                    feedback_items.extend(category_feedback)
        
        # Process patterns if available
        if "patterns" in analysis and analysis["patterns"]:
            for pattern in analysis["patterns"]:
                pattern_feedback = self._generate_pattern_feedback(pattern, analysis)
                if pattern_feedback:
                    feedback_items.append(pattern_feedback)
        
        # Prioritize feedback items
        prioritized = self._prioritize_feedback(feedback_items)
        
        # Limit to max_suggestions
        result = prioritized[:max_suggestions]
        
        logger.debug(f"Generated {len(result)} feedback suggestions")
        return result
    
    def map_feedback_to_actions(self, feedback_items: List[Dict[str, Any]]) -> List[Action]:
        """
        Map feedback suggestions to concrete actions.
        
        Args:
            feedback_items: List of feedback dictionaries.
            
        Returns:
            List of actions to implement the feedback.
        """
        actions = []
        
        for feedback in feedback_items:
            if "action_mapping" in feedback and feedback["action_mapping"]:
                # Extract action parameters
                action_type = feedback["action_mapping"]["action_type"]
                parameters = feedback["action_mapping"]["parameters"]
                
                # Create the action
                action = create_action(action_type, parameters=parameters)
                actions.append(action)
        
        logger.debug(f"Mapped {len(actions)} actions from feedback")
        return actions
    
    def _generate_semantic_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for semantic errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        feedback.append({
            "type": "semantic_clarification",
            "description": "Clarify the task requirements to avoid misunderstandings",
            "suggestion": "Provide a clearer explanation of what the task involves",
            "impact": "Reduces semantic misunderstandings",
            "action_mapping": {
                "action_type": "add_explanation",
                "parameters": {
                    "explanation_text": "Make sure to understand the exact requirements of the task before proceeding",
                    "target": "task"
                }
            }
        })
        
        feedback.append({
            "type": "task_examples",
            "description": "Include examples to demonstrate the expected behavior",
            "suggestion": "Add concrete examples of input and expected output",
            "impact": "Helps model understand the task through examples",
            "action_mapping": {
                "action_type": "add_example",
                "parameters": {
                    "example_text": "Input: [sample input]\nOutput: [expected output]",
                    "example_type": "input_output"
                }
            }
        })
        
        return feedback
    
    def _generate_format_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for format errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        feedback.append({
            "type": "format_specification",
            "description": "Specify the expected output format more precisely",
            "suggestion": "Provide a clear template for the output format",
            "impact": "Ensures consistent and correct formatting",
            "action_mapping": {
                "action_type": "specify_format",
                "parameters": {
                    "format_text": "Ensure your output follows this exact format: [format details]"
                }
            }
        })
        
        feedback.append({
            "type": "format_example",
            "description": "Include examples that demonstrate the correct format",
            "suggestion": "Add an example showing the exact required format",
            "impact": "Provides a concrete reference for the expected format",
            "action_mapping": {
                "action_type": "add_template",
                "parameters": {
                    "template_text": "[Detailed output template]",
                    "template_type": "structure"
                }
            }
        })
        
        return feedback
    
    def _generate_reasoning_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for reasoning errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        feedback.append({
            "type": "step_by_step_guidance",
            "description": "Break down the reasoning process into explicit steps",
            "suggestion": "Modify the workflow to include more detailed reasoning steps",
            "impact": "Improves logical reasoning by making steps explicit",
            "action_mapping": {
                "action_type": "modify_workflow",
                "parameters": {
                    "steps": [
                        "Analyze the input carefully",
                        "Break down the problem into components",
                        "Address each component systematically",
                        "Verify the logical consistency of your reasoning",
                        "Formulate the final response"
                    ]
                }
            }
        })
        
        return feedback
    
    def _generate_omission_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for omission errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        feedback.append({
            "type": "completeness_check",
            "description": "Add a step to verify completeness of the response",
            "suggestion": "Explicitly check that all required information is included",
            "impact": "Reduces omissions by adding a verification step",
            "action_mapping": {
                "action_type": "add_constraint",
                "parameters": {
                    "constraint_text": "Ensure your response includes all required information and addresses all aspects of the question"
                }
            }
        })
        
        return feedback
    
    def _generate_hallucination_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for hallucination errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        feedback.append({
            "type": "fact_checking",
            "description": "Add instructions to avoid making up information",
            "suggestion": "Explicitly instruct to only use information provided in the input",
            "impact": "Reduces hallucinations by emphasizing factual accuracy",
            "action_mapping": {
                "action_type": "add_rule",
                "parameters": {
                    "rule_text": "Only use information explicitly provided in the input. Do not make up or infer facts not directly supported by the given information.",
                    "priority": "high"
                }
            }
        })
        
        return feedback
    
    def _generate_boundary_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for boundary errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        feedback.append({
            "type": "scope_clarification",
            "description": "Clarify the boundaries of the task",
            "suggestion": "Explicitly define what should and should not be included",
            "impact": "Reduces boundary confusion by setting clear limits",
            "action_mapping": {
                "action_type": "add_constraint",
                "parameters": {
                    "constraint_text": "Only process the specific content provided between the delimiters. Do not extend analysis beyond the given text."
                }
            }
        })
        
        return feedback
    
    def _generate_context_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for context errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        feedback.append({
            "type": "context_emphasis",
            "description": "Emphasize the importance of considering all context",
            "suggestion": "Add instructions to fully consider the provided context",
            "impact": "Reduces context omissions by highlighting its importance",
            "action_mapping": {
                "action_type": "add_explanation",
                "parameters": {
                    "explanation_text": "Carefully consider all the context provided before formulating your response. Context is critical for accurate understanding.",
                    "target": "task"
                }
            }
        })
        
        return feedback
    
    def _generate_domain_feedback(
        self, 
        errors: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for domain knowledge errors.
        
        Args:
            errors: List of error information dictionaries.
            analysis: Complete error analysis.
            
        Returns:
            List of feedback dictionaries.
        """
        feedback = []
        
        # Generic domain knowledge feedback
        feedback.append({
            "type": "domain_knowledge_addition",
            "description": "Add relevant domain knowledge to the prompt",
            "suggestion": "Include key domain concepts and terminology",
            "impact": "Fills knowledge gaps by providing necessary domain context",
            "action_mapping": {
                "action_type": "add_domain_knowledge",
                "parameters": {
                    "knowledge_text": "Apply domain-specific knowledge and terminology appropriate for this task",
                    "domain": "general"
                }
            }
        })
        
        # More specific feedback could be added based on the actual domain
        # This would require domain detection logic
        
        return feedback
    
    def _generate_pattern_feedback(
        self, 
        pattern: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate feedback for an identified error pattern.
        
        Args:
            pattern: Error pattern dictionary.
            analysis: Complete error analysis.
            
        Returns:
            Feedback dictionary or None.
        """
        pattern_type = pattern.get("pattern_type")
        
        if pattern_type == "format_inconsistency":
            return {
                "type": "format_standardization",
                "description": "Standardize the output format to ensure consistency",
                "suggestion": "Provide a detailed output structure template",
                "impact": "Ensures consistent formatting across all responses",
                "action_mapping": {
                    "action_type": "structure_output",
                    "parameters": {
                        "structure_type": "sections",
                        "elements": ["section1", "section2", "section3"]
                    }
                }
            }
        elif pattern_type == "consistent_omission":
            return {
                "type": "checklist_addition",
                "description": "Add a checklist of required elements",
                "suggestion": "Include a verification step with an explicit checklist",
                "impact": "Prevents omissions by requiring explicit verification",
                "action_mapping": {
                    "action_type": "add_constraint",
                    "parameters": {
                        "constraint_text": "Before finalizing your response, verify that it includes all of these elements: [element1], [element2], [element3]"
                    }
                }
            }
        elif pattern_type == "hallucination_tendency":
            return {
                "type": "source_strict_adherence",
                "description": "Strictly limit responses to source information",
                "suggestion": "Add explicit instructions against adding unsupported information",
                "impact": "Reduces hallucinations by restricting to source information",
                "action_mapping": {
                    "action_type": "add_rule",
                    "parameters": {
                        "rule_text": "IMPORTANT: Do not add any information that is not explicitly stated in the source material. If uncertain, indicate that the information is not available.",
                        "priority": "high"
                    }
                }
            }
        elif pattern_type == "task_confusion":
            return {
                "type": "role_expertise",
                "description": "Strengthen the role with specific expertise",
                "suggestion": "Define a clearer expert role with relevant expertise",
                "impact": "Reduces confusion by establishing clear expertise and focus",
                "action_mapping": {
                    "action_type": "add_role",
                    "parameters": {
                        "role_text": "You are an expert specifically trained for this exact task, with deep understanding of all its requirements and nuances."
                    }
                }
            }
        elif pattern_type == "domain_knowledge_gap":
            return {
                "type": "terminology_clarification",
                "description": "Add definitions for domain-specific terminology",
                "suggestion": "Include a glossary of key domain terms",
                "impact": "Bridges knowledge gaps by providing necessary definitions",
                "action_mapping": {
                    "action_type": "clarify_terminology",
                    "parameters": {
                        "term": "[domain term]",
                        "definition": "[domain term definition]"
                    }
                }
            }
        
        return None
    
    def _prioritize_feedback(self, feedback_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize feedback items based on potential impact.
        
        Args:
            feedback_items: List of feedback dictionaries.
            
        Returns:
            Prioritized list of feedback dictionaries.
        """
        # Define priority scores for different feedback types
        priority_scores = {
            "semantic_clarification": 0.9,
            "format_specification": 0.8,
            "step_by_step_guidance": 0.8,
            "fact_checking": 0.9,
            "domain_knowledge_addition": 0.7,
            "completeness_check": 0.7,
            "scope_clarification": 0.6,
            "context_emphasis": 0.6,
            "format_standardization": 0.8,
            "checklist_addition": 0.7,
            "source_strict_adherence": 0.9,
            "role_expertise": 0.7,
            "terminology_clarification": 0.6,
            "task_examples": 0.7,
            "format_example": 0.7
        }
        
        # Score each feedback item
        scored_items = []
        for item in feedback_items:
            score = priority_scores.get(item.get("type"), 0.5)
            scored_items.append((item, score))
        
        # Sort by score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Return prioritized items
        return [item for item, _ in scored_items]