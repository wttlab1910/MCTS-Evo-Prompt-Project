"""
Knowledge base management module.

This module provides functionality for storing, retrieving, and managing
domain knowledge in the system.
"""
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import os
import json
import re
import datetime
from pathlib import Path

from app.utils.logger import get_logger
from app.utils.serialization import to_json, from_json, save_json, load_json
from app.utils.cache import MemoryCache
from app.config import DOMAIN_KNOWLEDGE_DIR, ERROR_PATTERNS_DIR

logger = get_logger("knowledge.knowledge_base")

class KnowledgeBase:
    """
    Knowledge base for storing domain knowledge.
    
    Provides functionality to store, retrieve, and manage knowledge items.
    """
    
    def __init__(self, domain_knowledge_dir: Optional[Path] = None, 
                 error_patterns_dir: Optional[Path] = None):
        """
        Initialize a knowledge base.
        
        Args:
            domain_knowledge_dir: Directory for domain knowledge storage.
            error_patterns_dir: Directory for error pattern storage.
        """
        self.domain_knowledge_dir = domain_knowledge_dir or DOMAIN_KNOWLEDGE_DIR
        self.error_patterns_dir = error_patterns_dir or ERROR_PATTERNS_DIR
        
        # Ensure directories exist
        self.domain_knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.error_patterns_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache for frequently accessed knowledge
        self.cache = MemoryCache(expiration=3600)  # 1 hour cache
        
        # Initialize domain index
        self._init_domain_index()
        
    def _init_domain_index(self):
        """Initialize and load domain index."""
        self.domain_index = {}
        self.domain_stats = {}
        
        # Load domain files
        for domain_file in self.domain_knowledge_dir.glob("*.json"):
            domain_name = domain_file.stem
            try:
                # Load knowledge items for this domain
                domain_data = load_json(domain_file)
                
                # Store metadata in index
                self.domain_index[domain_name] = {
                    "file": domain_file,
                    "count": len(domain_data),
                    "last_updated": self._get_file_mtime(domain_file)
                }
                
                # Store type statistics
                type_counts = {}
                for item in domain_data:
                    k_type = item.get("type", "unknown")
                    type_counts[k_type] = type_counts.get(k_type, 0) + 1
                    
                self.domain_stats[domain_name] = type_counts
                
                # Cache domain data
                cache_key = f"domain:{domain_name}"
                self.cache.set(cache_key, domain_data)
                
            except Exception as e:
                logger.error(f"Error loading domain '{domain_name}': {e}")
        
        logger.debug(f"Loaded {len(self.domain_index)} domains into knowledge base index")
    
    def _get_file_mtime(self, file_path: Path) -> str:
        """Get file modification time as ISO format string."""
        if not file_path.exists():
            return datetime.datetime.now().isoformat()
            
        mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
        return mtime.isoformat()
    
    def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Add a knowledge item to the knowledge base.
        
        Args:
            knowledge: Knowledge item to add.
            
        Returns:
            ID of the added knowledge item.
        """
        # Ensure the knowledge item has required fields
        if "id" not in knowledge:
            raise ValueError("Knowledge item must have an ID")
            
        if "type" not in knowledge:
            raise ValueError("Knowledge item must have a type")
            
        if "statement" not in knowledge:
            raise ValueError("Knowledge item must have a statement")
            
        # Get domain from metadata
        domain = knowledge.get("metadata", {}).get("domain", "general")
        
        # Load domain data
        domain_data = self.get_domain_knowledge(domain)
        
        # Check for duplicate ID
        existing_ids = {item.get("id") for item in domain_data}
        if knowledge["id"] in existing_ids:
            raise ValueError(f"Knowledge item with ID '{knowledge['id']}' already exists")
            
        # Add knowledge item
        domain_data.append(knowledge)
        
        # Save domain data
        self._save_domain_knowledge(domain, domain_data)
        
        # Update index and cache
        self.domain_index[domain] = {
            "file": self.domain_knowledge_dir / f"{domain}.json",
            "count": len(domain_data),
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Update type statistics
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {}
            
        k_type = knowledge.get("type", "unknown")
        self.domain_stats[domain][k_type] = self.domain_stats[domain].get(k_type, 0) + 1
        
        # Update cache
        cache_key = f"domain:{domain}"
        self.cache.set(cache_key, domain_data)
        
        logger.debug(f"Added knowledge item '{knowledge['id']}' to domain '{domain}'")
        return knowledge["id"]
    
    def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a knowledge item by ID.
        
        Args:
            knowledge_id: ID of the knowledge item.
            
        Returns:
            Knowledge item or None if not found.
        """
        # Check each domain for the knowledge item
        for domain in self.domain_index:
            domain_data = self.get_domain_knowledge(domain)
            for item in domain_data:
                if item.get("id") == knowledge_id:
                    return item
                    
        return None
    
    def update_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """
        Update a knowledge item.
        
        Args:
            knowledge: Updated knowledge item.
            
        Returns:
            True if successful, False otherwise.
        """
        knowledge_id = knowledge.get("id")
        if not knowledge_id:
            return False
            
        # Find the domain containing this knowledge item
        for domain in self.domain_index:
            domain_data = self.get_domain_knowledge(domain)
            for i, item in enumerate(domain_data):
                if item.get("id") == knowledge_id:
                    # Update the item
                    domain_data[i] = knowledge
                    
                    # Save domain data
                    self._save_domain_knowledge(domain, domain_data)
                    
                    # Update cache
                    cache_key = f"domain:{domain}"
                    self.cache.set(cache_key, domain_data)
                    
                    logger.debug(f"Updated knowledge item '{knowledge_id}' in domain '{domain}'")
                    return True
                    
        logger.warning(f"Knowledge item '{knowledge_id}' not found for update")
        return False
    
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete a knowledge item.
        
        Args:
            knowledge_id: ID of the knowledge item to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        # Find the domain containing this knowledge item
        for domain in self.domain_index:
            domain_data = self.get_domain_knowledge(domain)
            for i, item in enumerate(domain_data):
                if item.get("id") == knowledge_id:
                    # Remove the item
                    removed_item = domain_data.pop(i)
                    
                    # Save domain data
                    self._save_domain_knowledge(domain, domain_data)
                    
                    # Update index
                    self.domain_index[domain]["count"] = len(domain_data)
                    self.domain_index[domain]["last_updated"] = datetime.datetime.now().isoformat()
                    
                    # Update type statistics
                    k_type = removed_item.get("type", "unknown")
                    self.domain_stats[domain][k_type] = max(0, self.domain_stats[domain].get(k_type, 0) - 1)
                    
                    # Update cache
                    cache_key = f"domain:{domain}"
                    self.cache.set(cache_key, domain_data)
                    
                    logger.debug(f"Deleted knowledge item '{knowledge_id}' from domain '{domain}'")
                    return True
                    
        logger.warning(f"Knowledge item '{knowledge_id}' not found for deletion")
        return False
    
    def get_domain_knowledge(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all knowledge items for a domain.
        
        Args:
            domain: Domain name.
            
        Returns:
            List of knowledge items.
        """
        # Check cache first
        cache_key = f"domain:{domain}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Load from file
        domain_file = self.domain_knowledge_dir / f"{domain}.json"
        if domain_file.exists():
            try:
                domain_data = load_json(domain_file)
                
                # Update cache
                self.cache.set(cache_key, domain_data)
                
                return domain_data
            except Exception as e:
                logger.error(f"Error loading domain '{domain}': {e}")
                return []
        else:
            # Domain doesn't exist yet
            return []
    
    def search_knowledge(self, query: str, domains: List[str] = None, 
                         types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for knowledge items.
        
        Args:
            query: Search query.
            domains: List of domains to search (None for all).
            types: List of knowledge types to include (None for all).
            limit: Maximum number of results.
            
        Returns:
            List of matching knowledge items.
        """
        results = []
        
        # Normalize query
        query = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query))
        
        # Determine domains to search
        search_domains = domains if domains else list(self.domain_index.keys())
        
        # Search in each domain
        for domain in search_domains:
            domain_data = self.get_domain_knowledge(domain)
            
            for item in domain_data:
                # Filter by type if specified
                if types and item.get("type") not in types:
                    continue
                    
                # Match against statement
                statement = item.get("statement", "").lower()
                statement_terms = set(re.findall(r'\b\w+\b', statement))
                
                # Calculate term overlap
                overlap = len(query_terms.intersection(statement_terms))
                if overlap > 0:
                    # Match found, add to results with score
                    score = overlap / max(len(query_terms), 1)
                    results.append((item, score))
                    continue
                
                # Match against entities
                entities = item.get("entities", [])
                for entity in entities:
                    if entity.lower() in query:
                        results.append((item, 0.8))  # High score for entity match
                        break
        
        # Sort by score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in results[:limit]]
    
    def _save_domain_knowledge(self, domain: str, data: List[Dict[str, Any]]) -> bool:
        """
        Save domain knowledge to file.
        
        Args:
            domain: Domain name.
            data: Knowledge data to save.
            
        Returns:
            True if successful, False otherwise.
        """
        domain_file = self.domain_knowledge_dir / f"{domain}.json"
        try:
            save_json(data, domain_file)
            return True
        except Exception as e:
            logger.error(f"Error saving domain '{domain}': {e}")
            return False
    
    def query_by_entities(self, entities: List[str], domains: List[str] = None, 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query knowledge by entities.
        
        Args:
            entities: List of entities to search for.
            domains: List of domains to search (None for all).
            limit: Maximum number of results.
            
        Returns:
            List of matching knowledge items.
        """
        results = []
        
        # Normalize entities
        entity_set = {e.lower() for e in entities if e}
        
        # Determine domains to search
        search_domains = domains if domains else list(self.domain_index.keys())
        
        # Search in each domain
        for domain in search_domains:
            domain_data = self.get_domain_knowledge(domain)
            
            for item in domain_data:
                # Match against item entities
                item_entities = {e.lower() for e in item.get("entities", [])}
                
                # Calculate entity overlap
                overlap = len(entity_set.intersection(item_entities))
                if overlap > 0:
                    # Match found, add to results with score
                    score = overlap / max(len(entity_set), 1)
                    results.append((item, score))
                    continue
                
                # Check relations for entity matches
                relations = item.get("relations", [])
                for relation in relations:
                    subject = relation.get("subject", "").lower()
                    obj = relation.get("object", "").lower()
                    
                    if subject in entity_set or obj in entity_set:
                        results.append((item, 0.7))  # Good score for relation match
                        break
        
        # Sort by score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in results[:limit]]
    
    def get_domains(self) -> List[Dict[str, Any]]:
        """
        Get list of domains with statistics.
        
        Returns:
            List of domain information.
        """
        domains = []
        
        for domain, info in self.domain_index.items():
            domain_info = {
                "name": domain,
                "count": info["count"],
                "last_updated": info["last_updated"],
                "types": self.domain_stats.get(domain, {})
            }
            domains.append(domain_info)
            
        return domains
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary with statistics.
        """
        total_items = sum(info["count"] for info in self.domain_index.values())
        
        # Aggregate type counts across domains
        type_counts = {}
        for domain, type_data in self.domain_stats.items():
            for k_type, count in type_data.items():
                type_counts[k_type] = type_counts.get(k_type, 0) + count
        
        stats = {
            "total_items": total_items,
            "domains": len(self.domain_index),
            "types": type_counts
        }
        
        return stats
    
    def add_error_pattern(self, pattern: Dict[str, Any]) -> str:
        """
        Add an error pattern to the knowledge base.
        
        Args:
            pattern: Error pattern to add.
            
        Returns:
            ID of the added pattern.
        """
        # Ensure the pattern has required fields
        if "id" not in pattern:
            raise ValueError("Error pattern must have an ID")
            
        if "pattern_type" not in pattern:
            raise ValueError("Error pattern must have a type")
            
        if "description" not in pattern:
            raise ValueError("Error pattern must have a description")
            
        # Get task type from metadata
        task_type = pattern.get("task_type", "general")
        
        # Load task type patterns
        patterns_file = self.error_patterns_dir / f"{task_type}.json"
        if patterns_file.exists():
            try:
                patterns_data = load_json(patterns_file)
            except Exception as e:
                logger.error(f"Error loading patterns for '{task_type}': {e}")
                patterns_data = []
        else:
            patterns_data = []
        
        # Check for duplicate ID
        existing_ids = {p.get("id") for p in patterns_data}
        if pattern["id"] in existing_ids:
            raise ValueError(f"Error pattern with ID '{pattern['id']}' already exists")
            
        # Add pattern
        patterns_data.append(pattern)
        
        # Save patterns data
        try:
            save_json(patterns_data, patterns_file)
            logger.debug(f"Added error pattern '{pattern['id']}' to task type '{task_type}'")
            return pattern["id"]
        except Exception as e:
            logger.error(f"Error saving patterns for '{task_type}': {e}")
            raise
    
    def get_error_patterns(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Get error patterns for a task type.
        
        Args:
            task_type: Task type.
            
        Returns:
            List of error patterns.
        """
        # Load task type patterns
        patterns_file = self.error_patterns_dir / f"{task_type}.json"
        if patterns_file.exists():
            try:
                return load_json(patterns_file)
            except Exception as e:
                logger.error(f"Error loading patterns for '{task_type}': {e}")
                return []
        else:
            return []