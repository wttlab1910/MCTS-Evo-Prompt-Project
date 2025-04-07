"""
Error analysis for prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple, Counter
from collections import defaultdict, Counter
import re
from app.utils.logger import get_logger

logger = get_logger("error.analyzer")

class ErrorAnalyzer:
    """
    Analyzes errors to identify patterns and root causes.
    
    This class handles clustering errors by type, identifying common patterns,
    and generating structured error descriptions.
    """
    
    def __init__(self):
        """Initialize an error analyzer."""
        # Error categories and their descriptions
        self.error_categories = {
            "semantic_error": "Misunderstanding of the task requirements",
            "format_error": "Incorrect output format or structure",
            "reasoning_error": "Flawed reasoning or logical process",
            "omission_error": "Missing required information in the output",
            "hallucination_error": "Including incorrect or made-up information",
            "boundary_error": "Confusion about where to start or end processing",
            "context_error": "Failing to consider the provided context",
            "domain_error": "Lack of domain-specific knowledge"
        }
        
        # Keywords associated with each error category
        self.error_keywords = {
            "semantic_error": ["misunderstood", "misinterpreted", "wrong meaning", "incorrect interpretation"],
            "format_error": ["format", "structure", "layout", "organization"],
            "reasoning_error": ["logic", "reasoning", "deduction", "inference", "conclusion"],
            "omission_error": ["missing", "omitted", "excluded", "left out"],
            "hallucination_error": ["fabricated", "made up", "invented", "incorrect fact"],
            "boundary_error": ["boundary", "scope", "limit", "extend"],
            "context_error": ["context", "surrounding", "related", "background"],
            "domain_error": ["domain", "field", "specialty", "expertise", "technical"]
        }
        
        logger.debug("Initialized ErrorAnalyzer")
    
    def analyze_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a collection of errors to identify patterns.
        
        Args:
            errors: List of error information dictionaries.
            
        Returns:
            Dictionary with analysis results, including error clusters and patterns.
        """
        if not errors:
            logger.warning("No errors to analyze")
            return {"error_clusters": {}, "patterns": [], "summary": "No errors to analyze"}
        
        # Categorize errors
        error_clusters = self._categorize_errors(errors)
        
        # Identify patterns
        patterns = self._identify_patterns(errors, error_clusters)
        
        # Generate summary
        summary = self._generate_summary(error_clusters, patterns)
        
        analysis = {
            "error_clusters": error_clusters,
            "patterns": patterns,
            "summary": summary
        }
        
        logger.debug(f"Analyzed {len(errors)} errors, found {len(patterns)} patterns")
        return analysis
    
    def _categorize_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize errors by type.
        
        Args:
            errors: List of error information dictionaries.
            
        Returns:
            Dictionary mapping error categories to lists of errors.
        """
        clusters = defaultdict(list)
        
        for error in errors:
            # If error_type is already a known category, use it
            if "error_type" in error and error["error_type"] in self.error_categories:
                category = error["error_type"]
            else:
                # Otherwise, determine the category based on error content
                category = self._determine_error_category(error)
            
            clusters[category].append(error)
        
        # Convert defaultdict to regular dict for return
        return dict(clusters)
    
    def _determine_error_category(self, error: Dict[str, Any]) -> str:
        """
        Determine the category of an error based on its content.
        
        Args:
            error: Error information dictionary.
            
        Returns:
            Error category string.
        """
        # Extract text from error
        error_text = ""
        for field in ["actual", "error_message"]:
            if field in error and error[field]:
                error_text += str(error[field]) + " "
        
        # Check for keywords in each category
        category_scores = {}
        for category, keywords in self.error_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in error_text.lower())
            category_scores[category] = score
        
        # Get category with highest score
        if category_scores:
            max_score = max(category_scores.values())
            if max_score > 0:
                # Find all categories with the max score
                max_categories = [cat for cat, score in category_scores.items() if score == max_score]
                # If multiple categories have the same score, choose one
                return max_categories[0]
        
        # Default to "semantic_error" if no clear category
        return "semantic_error"
    
    def _identify_patterns(
        self, 
        errors: List[Dict[str, Any]], 
        clusters: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Identify common patterns in errors.
        
        Args:
            errors: List of all error information dictionaries.
            clusters: Dictionary mapping error categories to lists of errors.
            
        Returns:
            List of pattern dictionaries, each describing a common error pattern.
        """
        patterns = []
        
        # Check for format-related errors
        if "format_error" in clusters and len(clusters["format_error"]) >= 2:
            patterns.append({
                "pattern_type": "format_inconsistency",
                "description": "Inconsistent output format across examples",
                "frequency": len(clusters["format_error"]),
                "examples": [e["example_id"] for e in clusters["format_error"][:3]]
            })
        
        # Check for omission errors
        if "omission_error" in clusters and len(clusters["omission_error"]) >= 2:
            patterns.append({
                "pattern_type": "consistent_omission",
                "description": "Consistently omitting required information",
                "frequency": len(clusters["omission_error"]),
                "examples": [e["example_id"] for e in clusters["omission_error"][:3]]
            })
        
        # Check for hallucination errors
        if "hallucination_error" in clusters and len(clusters["hallucination_error"]) >= 1:
            patterns.append({
                "pattern_type": "hallucination_tendency",
                "description": "Tendency to generate incorrect or unsupported information",
                "frequency": len(clusters["hallucination_error"]),
                "examples": [e["example_id"] for e in clusters["hallucination_error"][:3]]
            })
        
        # Check for multiple error types (confusion)
        if len(clusters) >= 3:
            patterns.append({
                "pattern_type": "task_confusion",
                "description": "Multiple error types suggest general confusion about the task",
                "frequency": len(errors),
                "categories": list(clusters.keys())
            })
        
        # Check for domain-specific errors
        if "domain_error" in clusters:
            patterns.append({
                "pattern_type": "domain_knowledge_gap",
                "description": "Lack of necessary domain-specific knowledge",
                "frequency": len(clusters["domain_error"]),
                "examples": [e["example_id"] for e in clusters["domain_error"][:3]]
            })
        
        return patterns
    
    def _generate_summary(
        self, 
        clusters: Dict[str, List[Dict[str, Any]]], 
        patterns: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a summary of the error analysis.
        
        Args:
            clusters: Dictionary mapping error categories to lists of errors.
            patterns: List of identified error patterns.
            
        Returns:
            Summary string.
        """
        if not clusters:
            return "No errors detected."
        
        # Count errors by category
        category_counts = {category: len(errors) for category, errors in clusters.items()}
        total_errors = sum(category_counts.values())
        
        # Identify primary error category
        primary_category = max(category_counts.items(), key=lambda x: x[1])[0]
        
        # Generate summary text
        summary = f"Analysis of {total_errors} errors reveals that "
        
        if total_errors == 1:
            summary = f"Analysis of a single error indicates a {primary_category}. "
        elif len(clusters) == 1:
            summary += f"all errors are of type {primary_category}. "
        else:
            summary += f"the primary issue is {primary_category} ({category_counts[primary_category]} errors), "
            secondary_categories = sorted(
                [(c, n) for c, n in category_counts.items() if c != primary_category],
                key=lambda x: x[1],
                reverse=True
            )
            if secondary_categories:
                secondary = secondary_categories[0]
                summary += f"followed by {secondary[0]} ({secondary[1]} errors). "
        
        # Add pattern information
        if patterns:
            pattern_desc = patterns[0]["description"].lower()
            summary += f"The main pattern observed is {pattern_desc}. "
            
            if len(patterns) > 1:
                summary += f"Additionally, {len(patterns) - 1} other patterns were identified."
        
        return summary
    
    def get_error_category_description(self, category: str) -> str:
        """
        Get the description for an error category.
        
        Args:
            category: Error category string.
            
        Returns:
            Description of the error category.
        """
        return self.error_categories.get(
            category, 
            "Unknown error type"
        )
    
    def add_error_category(self, category: str, description: str, keywords: List[str]) -> None:
        """
        Add a new error category to the analyzer.
        
        Args:
            category: Error category string.
            description: Description of the error category.
            keywords: List of keywords associated with the category.
        """
        self.error_categories[category] = description
        self.error_keywords[category] = keywords
        logger.debug(f"Added new error category: {category}")