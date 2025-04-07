"""
Knowledge service for managing domain knowledge.
"""
from typing import Dict, Any, List, Optional, Union
import asyncio
import uuid
import json
import time
from datetime import datetime
from pathlib import Path
from app.config import DOMAIN_KNOWLEDGE_DIR, ERROR_PATTERNS_DIR
from app.utils.logger import get_logger
from app.utils.timer import Timer
from app.utils.serialization import save_json, load_json

logger = get_logger("services.knowledge_service")

class KnowledgeService:
    """
    Service for managing domain knowledge.
    
    This service handles domain knowledge extraction, verification, and integration.
    """
    
    def __init__(self):
        """Initialize the knowledge service."""
        # Create directories if they don't exist
        DOMAIN_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        ERROR_PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing knowledge
        self._load_knowledge()
        
        logger.info("Knowledge service initialized.")
    
    def _load_knowledge(self):
        """Load existing knowledge from files."""
        self.domain_knowledge = {}
        self.error_patterns = {}
        
        try:
            # Load domain knowledge
            for file_path in DOMAIN_KNOWLEDGE_DIR.glob("*.json"):
                try:
                    entry = load_json(file_path)
                    self.domain_knowledge[file_path.stem] = entry
                except Exception as e:
                    logger.error(f"Failed to load domain knowledge from {file_path}: {e}")
            
            # Load error patterns
            for file_path in ERROR_PATTERNS_DIR.glob("*.json"):
                try:
                    entry = load_json(file_path)
                    self.error_patterns[file_path.stem] = entry
                except Exception as e:
                    logger.error(f"Failed to load error pattern from {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.domain_knowledge)} domain knowledge entries and {len(self.error_patterns)} error patterns")
        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")
    
    async def list_entries(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List knowledge entries, optionally filtered by domain.
        
        Args:
            domain: Domain filter (optional).
            
        Returns:
            List of knowledge entries.
        """
        entries = []
        
        # Add domain knowledge entries
        for entry_id, entry in self.domain_knowledge.items():
            if domain is None or entry.get("domain") == domain:
                entries.append({
                    "id": entry_id,
                    **entry
                })
        
        return entries
    
    async def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific knowledge entry.
        
        Args:
            entry_id: Knowledge entry ID.
            
        Returns:
            Knowledge entry or None if not found.
        """
        # Check domain knowledge
        if entry_id in self.domain_knowledge:
            return {
                "id": entry_id,
                **self.domain_knowledge[entry_id]
            }
        
        # Check error patterns
        if entry_id in self.error_patterns:
            return {
                "id": entry_id,
                **self.error_patterns[entry_id]
            }
        
        return None
    
    async def create_entry(self,
                         knowledge_type: str,
                         statement: str,
                         domain: str,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new knowledge entry.
        
        Args:
            knowledge_type: Type of knowledge entry.
            statement: Knowledge statement.
            domain: Knowledge domain.
            metadata: Additional metadata (optional).
            
        Returns:
            Created knowledge entry.
        """
        # Generate entry ID
        entry_id = str(uuid.uuid4())
        
        # Create entry
        entry = {
            "knowledge_type": knowledge_type,
            "statement": statement,
            "domain": domain,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": None
        }
        
        # Save entry
        if knowledge_type.startswith("error_"):
            # Save as error pattern
            self.error_patterns[entry_id] = entry
            save_json(entry, ERROR_PATTERNS_DIR / f"{entry_id}.json")
        else:
            # Save as domain knowledge
            self.domain_knowledge[entry_id] = entry
            save_json(entry, DOMAIN_KNOWLEDGE_DIR / f"{entry_id}.json")
        
        logger.info(f"Created knowledge entry {entry_id} of type {knowledge_type}")
        
        return {
            "id": entry_id,
            **entry
        }
    
    async def update_entry(self,
                         entry_id: str,
                         knowledge_type: str,
                         statement: str,
                         domain: str,
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Update a knowledge entry.
        
        Args:
            entry_id: Knowledge entry ID.
            knowledge_type: Type of knowledge entry.
            statement: Knowledge statement.
            domain: Knowledge domain.
            metadata: Additional metadata (optional).
            
        Returns:
            Updated knowledge entry or None if not found.
        """
        # Check if entry exists
        original_entry = None
        if entry_id in self.domain_knowledge:
            original_entry = self.domain_knowledge[entry_id]
        elif entry_id in self.error_patterns:
            original_entry = self.error_patterns[entry_id]
        
        if not original_entry:
            return None
        
        # Create updated entry
        updated_entry = {
            "knowledge_type": knowledge_type,
            "statement": statement,
            "domain": domain,
            "metadata": metadata or {},
            "created_at": original_entry["created_at"],
            "updated_at": datetime.now().isoformat()
        }
        
        # Save updated entry
        if knowledge_type.startswith("error_"):
            # Save as error pattern
            self.error_patterns[entry_id] = updated_entry
            save_json(updated_entry, ERROR_PATTERNS_DIR / f"{entry_id}.json")
            
            # Remove from domain knowledge if it was there
            if entry_id in self.domain_knowledge:
                self.domain_knowledge.pop(entry_id)
                domain_path = DOMAIN_KNOWLEDGE_DIR / f"{entry_id}.json"
                if domain_path.exists():
                    domain_path.unlink()
        else:
            # Save as domain knowledge
            self.domain_knowledge[entry_id] = updated_entry
            save_json(updated_entry, DOMAIN_KNOWLEDGE_DIR / f"{entry_id}.json")
            
            # Remove from error patterns if it was there
            if entry_id in self.error_patterns:
                self.error_patterns.pop(entry_id)
                error_path = ERROR_PATTERNS_DIR / f"{entry_id}.json"
                if error_path.exists():
                    error_path.unlink()
        
        logger.info(f"Updated knowledge entry {entry_id}")
        
        return {
            "id": entry_id,
            **updated_entry
        }
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry.
        
        Args:
            entry_id: Knowledge entry ID.
            
        Returns:
            True if deleted successfully, False otherwise.
        """
        # Check if entry exists in domain knowledge
        if entry_id in self.domain_knowledge:
            self.domain_knowledge.pop(entry_id)
            domain_path = DOMAIN_KNOWLEDGE_DIR / f"{entry_id}.json"
            if domain_path.exists():
                domain_path.unlink()
            logger.info(f"Deleted domain knowledge entry {entry_id}")
            return True
        
        # Check if entry exists in error patterns
        if entry_id in self.error_patterns:
            self.error_patterns.pop(entry_id)
            error_path = ERROR_PATTERNS_DIR / f"{entry_id}.json"
            if error_path.exists():
                error_path.unlink()
            logger.info(f"Deleted error pattern entry {entry_id}")
            return True
        
        return False