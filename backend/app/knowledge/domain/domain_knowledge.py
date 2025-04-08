"""
Domain knowledge management module.

This module provides functionality for storing, retrieving, and managing
domain-specific knowledge.
"""
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import os
import json
import re
from pathlib import Path

from app.utils.logger import get_logger
from app.knowledge.knowledge_base import KnowledgeBase
from app.knowledge.extraction.extractor import KnowledgeExtractor
from app.knowledge.extraction.verification import KnowledgeVerifier

logger = get_logger("knowledge.domain.domain_knowledge")

class DomainKnowledgeManager:
    """
    Manager for domain-specific knowledge.
    
    Provides functionality to extract, verify, and manage domain knowledge.
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize a domain knowledge manager.
        
        Args:
            knowledge_base: Knowledge base for storage (creates new one if None).
        """
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.extractors = self._init_extractors()
        self.verifiers = self._init_verifiers()
        
    def _init_extractors(self) -> Dict[str, KnowledgeExtractor]:
        """Initialize knowledge extractors."""
        from app.knowledge.extraction.extractor import (
            ErrorBasedExtractor,
            ConceptualKnowledgeExtractor,
            ProceduralKnowledgeExtractor
        )
        
        extractors = {
            "error": ErrorBasedExtractor(),
            "conceptual": ConceptualKnowledgeExtractor(),
            "procedural": ProceduralKnowledgeExtractor()
        }
        
        return extractors
    
    def _init_verifiers(self) -> Dict[str, KnowledgeVerifier]:
        """Initialize knowledge verifiers."""
        from app.knowledge.extraction.verification import (
            ConsistencyVerifier,
            RelationshipMapper,
            ConfidenceScorer
        )
        
        verifiers = {
            "consistency": ConsistencyVerifier(self.knowledge_base),
            "relationship": RelationshipMapper(),
            "confidence": ConfidenceScorer()
        }
        
        return verifiers
    
    def extract_knowledge(self, source: Any, extractor_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract knowledge from a source.
        
        Args:
            source: Source to extract knowledge from.
            extractor_type: Type of extractor to use.
            **kwargs: Additional extraction parameters.
                - domain: Optional domain to associate with extracted knowledge.
                
        Returns:
            List of extracted knowledge items.
        """
        if extractor_type not in self.extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
            
        extractor = self.extractors[extractor_type]
        
        # Extract knowledge
        extracted_items = extractor.extract(source, **kwargs)
        
        logger.debug(f"Extracted {len(extracted_items)} knowledge items using {extractor_type} extractor")
        return extracted_items
    
    def verify_knowledge(self, knowledge: Union[Dict[str, Any], List[Dict[str, Any]]], 
                         verify_types: List[str] = None, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Verify knowledge items.
        
        Args:
            knowledge: Knowledge item or list of items to verify.
            verify_types: Types of verification to perform (None for all).
            **kwargs: Additional verification parameters.
                
        Returns:
            Verified knowledge item(s).
        """
        # Convert single item to list for processing
        is_single = not isinstance(knowledge, list)
        items = [knowledge] if is_single else knowledge
        
        # Determine verification types
        if not verify_types:
            verify_types = list(self.verifiers.keys())
            
        # Verify each item with each verifier
        verified_items = items
        
        for verify_type in verify_types:
            if verify_type not in self.verifiers:
                logger.warning(f"Unknown verifier type: {verify_type}")
                continue
                
            verifier = self.verifiers[verify_type]
            verified_items = verifier.batch_verify(verified_items, **kwargs)
        
        # Return same type as input
        return verified_items[0] if is_single else verified_items
    
    def add_knowledge(self, knowledge: Union[Dict[str, Any], List[Dict[str, Any]]], 
                      verify: bool = True, extract_relationships: bool = True) -> List[str]:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge item or list of items to add.
            verify: Whether to verify knowledge before adding.
            extract_relationships: Whether to extract relationships.
            
        Returns:
            List of added knowledge IDs.
        """
        # Convert single item to list for processing
        items = [knowledge] if not isinstance(knowledge, list) else knowledge
        
        # Verify knowledge if requested
        if verify:
            verify_types = ["consistency", "confidence"]
            items = self.verify_knowledge(items, verify_types)
        
        # Extract relationships if requested
        if extract_relationships:
            items = self.verify_knowledge(items, ["relationship"])
        
        # Add each item to the knowledge base
        added_ids = []
        
        for item in items:
            try:
                item_id = self.knowledge_base.add_knowledge(item)
                added_ids.append(item_id)
            except Exception as e:
                logger.error(f"Error adding knowledge item: {e}")
        
        logger.debug(f"Added {len(added_ids)} knowledge items to the knowledge base")
        return added_ids
    
    def get_domain_knowledge(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all knowledge for a domain.
        
        Args:
            domain: Domain name.
            
        Returns:
            List of knowledge items.
        """
        return self.knowledge_base.get_domain_knowledge(domain)
    
    def query_knowledge(self, query: str = None, entities: List[str] = None, 
                       domains: List[str] = None, types: List[str] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.
        
        Args:
            query: Text query (None to skip).
            entities: Entity list to match (None to skip).
            domains: Domains to search (None for all).
            types: Knowledge types to include (None for all).
            limit: Maximum number of results.
            
        Returns:
            List of matching knowledge items.
        """
        results = []
        
        # Query by text if provided
        if query:
            text_results = self.knowledge_base.search_knowledge(
                query=query,
                domains=domains,
                types=types,
                limit=limit
            )
            results.extend(text_results)
        
        # Query by entities if provided
        if entities:
            entity_results = self.knowledge_base.query_by_entities(
                entities=entities,
                domains=domains,
                limit=limit
            )
            
            # Combine results, avoiding duplicates
            existing_ids = {item.get("id") for item in results}
            for item in entity_results:
                if item.get("id") not in existing_ids:
                    results.append(item)
                    existing_ids.add(item.get("id"))
        
        # Limit combined results
        return results[:limit]