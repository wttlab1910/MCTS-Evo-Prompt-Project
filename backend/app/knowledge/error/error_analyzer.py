"""
Error analysis module.

This module analyzes errors from model responses to identify patterns and root causes.
"""
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import re
from collections import defaultdict
import hashlib

from app.utils.logger import get_logger

logger = get_logger("knowledge.error.error_analyzer")

class ErrorAnalyzer:
    """
    Analyze errors to identify patterns and extract insights.
    
    This analyzer examines errors from model responses to identify recurring patterns,
    categorize them, and provide structured analysis.
    """
    
    def __init__(self):
        """Initialize an error analyzer."""
        # Initialize error categories and patterns
        self.error_categories = {
            "entity_confusion": "Confusion between different entity types or categories",
            "procedure_error": "Errors in following a sequence of steps or a process",
            "domain_misconception": "Misconceptions about domain-specific concepts",
            "format_inconsistency": "Inconsistencies in output format or structure",
            "boundary_confusion": "Confusion at boundary cases or special situations"
        }
        
    def analyze_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a list of errors to identify patterns.
        
        Args:
            errors: List of error dictionaries.
            
        Returns:
            Dictionary with analysis results, including:
            - error_clusters: Errors grouped by category
            - patterns: Identified error patterns
            - summary: Summary of analysis
        """
        if not errors:
            logger.warning("No errors to analyze")
            return {
                "error_clusters": {},
                "patterns": [],
                "summary": "No errors to analyze"
            }
        
        # Cluster errors by type
        error_clusters = self._cluster_errors(errors)
        
        # Extract patterns from clusters
        patterns = self._extract_patterns(error_clusters, errors)
        
        # Generate summary
        summary = self._generate_summary(error_clusters, patterns)
        
        analysis = {
            "error_clusters": error_clusters,
            "patterns": patterns,
            "summary": summary
        }
        
        logger.debug(f"Analyzed {len(errors)} errors, found {len(patterns)} patterns")
        return analysis
    
    def _cluster_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Cluster errors by type or category.
        
        Args:
            errors: List of error dictionaries.
            
        Returns:
            Dictionary mapping error categories to lists of error IDs.
        """
        clusters = defaultdict(list)
        
        for error in errors:
            error_id = error.get("example_id", str(hash(str(error))))
            error_type = error.get("error_type", "unknown")
            description = error.get("description", "")
            
            # Determine the most appropriate category
            category = self._categorize_error(error_type, description)
            
            # Add to cluster
            clusters[category].append(error_id)
        
        return dict(clusters)
    
    def _categorize_error(self, error_type: str, description: str) -> str:
        """
        Categorize an error based on type and description.
        
        Args:
            error_type: Error type string.
            description: Error description.
            
        Returns:
            Error category.
        """
        # First try to match based on error_type
        error_type_lower = error_type.lower()
        
        if "entity" in error_type_lower or "classification" in error_type_lower:
            return "entity_confusion"
        
        if "procedure" in error_type_lower or "step" in error_type_lower or "sequence" in error_type_lower:
            return "procedure_error"
        
        if "concept" in error_type_lower or "domain" in error_type_lower or "knowledge" in error_type_lower:
            return "domain_misconception"
        
        if "format" in error_type_lower or "structure" in error_type_lower or "output" in error_type_lower:
            return "format_inconsistency"
        
        if "boundary" in error_type_lower or "edge" in error_type_lower or "special" in error_type_lower:
            return "boundary_confusion"
        
        # If no match by type, try to match by description
        description_lower = description.lower()
        
        if ("confused" in description_lower or "mistook" in description_lower or 
            "classified as" in description_lower or "instead of" in description_lower):
            return "entity_confusion"
        
        if ("steps" in description_lower or "procedure" in description_lower or 
            "process" in description_lower or "sequence" in description_lower):
            return "procedure_error"
        
        if ("concept" in description_lower or "definition" in description_lower or 
            "meaning" in description_lower or "interpreted" in description_lower):
            return "domain_misconception"
        
        if ("format" in description_lower or "structure" in description_lower or 
            "layout" in description_lower or "presentation" in description_lower):
            return "format_inconsistency"
        
        if ("edge case" in description_lower or "boundary" in description_lower or 
            "special case" in description_lower or "exception" in description_lower):
            return "boundary_confusion"
        
        # Default to domain misconception if no better match
        return "domain_misconception"
    
    def _extract_patterns(self, clusters: Dict[str, List[str]], errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract patterns from error clusters.
        
        Args:
            clusters: Error clusters by category.
            errors: Original error dictionaries.
            
        Returns:
            List of identified patterns.
        """
        patterns = []
        
        # Create lookup for errors by ID
        error_lookup = {}
        for error in errors:
            error_id = error.get("example_id", str(hash(str(error))))
            error_lookup[error_id] = error
        
        # Process each cluster
        for category, error_ids in clusters.items():
            category_errors = [error_lookup[error_id] for error_id in error_ids if error_id in error_lookup]
            
            if not category_errors:
                continue
                
            # Group similar errors within the category
            subclusters = self._group_similar_errors(category_errors)
            
            # Create a pattern for each subcluster
            for subcluster in subclusters:
                if not subcluster:
                    continue
                    
                # Generate a pattern from the subcluster
                pattern = self._generate_pattern(category, subcluster)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _group_similar_errors(self, errors: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group similar errors together.
        
        Args:
            errors: List of error dictionaries.
            
        Returns:
            List of error subclusters.
        """
        if len(errors) <= 1:
            return [errors]
            
        subclusters = []
        remaining = errors.copy()
        
        while remaining:
            # Take first error as reference
            reference = remaining.pop(0)
            subcluster = [reference]
            
            # Find similar errors
            i = 0
            while i < len(remaining):
                if self._are_errors_similar(reference, remaining[i]):
                    subcluster.append(remaining.pop(i))
                else:
                    i += 1
            
            subclusters.append(subcluster)
        
        return subclusters
    
    def _are_errors_similar(self, error1: Dict[str, Any], error2: Dict[str, Any]) -> bool:
        """
        Check if two errors are similar.
        
        Args:
            error1: First error dictionary.
            error2: Second error dictionary.
            
        Returns:
            True if errors are similar, False otherwise.
        """
        # Same error type is a strong indicator
        if error1.get("error_type") == error2.get("error_type"):
            return True
            
        # Check description similarity
        desc1 = error1.get("description", "").lower()
        desc2 = error2.get("description", "").lower()
        
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', desc1))
        words2 = set(re.findall(r'\b\w+\b', desc2))
        
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / max(len(words1), len(words2))
        
        return similarity > 0.5
    
    def _generate_pattern(self, category: str, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a pattern from a group of similar errors.
        
        Args:
            category: Error category.
            errors: List of similar error dictionaries.
            
        Returns:
            Pattern dictionary.
        """
        if not errors:
            return None
            
        # Use the most detailed error description as base
        descriptions = [e.get("description", "") for e in errors]
        base_description = max(descriptions, key=len) if descriptions else ""
        
        # Extract entities mentioned
        entities = set()
        for error in errors:
            # Extract entities from example
            example = error.get("example", {})
            example_text = example.get("text", "")
            
            # Look for entities (capitalized words that aren't at start of sentence)
            for match in re.finditer(r'(?<!\.\s)(?<!\?\s)(?<!\!\s)\b([A-Z][a-zA-Z0-9]*)\b', example_text):
                entity = match.group(1)
                if len(entity) >= 2 and entity not in ["I", "A", "The"]:
                    entities.add(entity)
            
            # Look for entities in description
            for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]*)\b', error.get("description", "")):
                entity = match.group(1)
                if len(entity) >= 2 and entity not in ["I", "A", "The"]:
                    entities.add(entity)
        
        # Generate a pattern ID
        pattern_id = hashlib.md5(base_description.encode('utf-8')).hexdigest()[:8]
        
        # Create the pattern
        pattern = {
            "id": f"p_{pattern_id}",
            "pattern_type": category,
            "description": base_description,
            "entities": list(entities),
            "frequency": len(errors),
            "examples": [error.get("example_id", "") for error in errors if "example_id" in error]
        }
        
        return pattern
    
    def _generate_summary(self, clusters: Dict[str, List[str]], patterns: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the error analysis.
        
        Args:
            clusters: Error clusters by category.
            patterns: Identified patterns.
            
        Returns:
            Summary text.
        """
        if not patterns:
            return "No clear error patterns identified."
            
        # Count errors by category
        category_counts = {category: len(error_ids) for category, error_ids in clusters.items()}
        
        # Find the most common category
        top_category = max(category_counts.items(), key=lambda x: x[1]) if category_counts else (None, 0)
        
        # Generate summary
        summary_parts = []
        
        # Overview
        total_errors = sum(category_counts.values())
        summary_parts.append(f"Analyzed {total_errors} errors and identified {len(patterns)} distinct patterns.")
        
        # Top category
        if top_category[0]:
            category_desc = self.error_categories.get(top_category[0], top_category[0])
            summary_parts.append(f"Most common error type: {category_desc} ({top_category[1]} instances).")
        
        # Top patterns
        if patterns:
            top_pattern = max(patterns, key=lambda p: p.get("frequency", 0))
            summary_parts.append(f"Top pattern: {top_pattern['description'][:100]}... ({top_pattern['frequency']} instances).")
        
        return " ".join(summary_parts)
    
    def get_error_category_description(self, category: str) -> str:
        """
        Get the description for an error category.
        
        Args:
            category: Error category.
            
        Returns:
            Category description.
        """
        return self.error_categories.get(category, category)