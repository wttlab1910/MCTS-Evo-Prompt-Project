"""
Tests for the Knowledge Base functionality.

Tests the storage, retrieval, and management of knowledge items in the knowledge base.
"""
import pytest
import os
import tempfile
from pathlib import Path
from app.knowledge.knowledge_base import KnowledgeBase
from app.knowledge.domain.domain_knowledge import DomainKnowledgeManager

class TestKnowledgeBase:
    """Test cases for knowledge base functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for the knowledge base
        self.temp_dir = tempfile.TemporaryDirectory()
        self.domain_dir = Path(self.temp_dir.name) / "domain_knowledge"
        self.error_dir = Path(self.temp_dir.name) / "error_patterns"
        
        self.domain_dir.mkdir(exist_ok=True)
        self.error_dir.mkdir(exist_ok=True)
        
        # Initialize knowledge base with temp directories
        self.kb = KnowledgeBase(self.domain_dir, self.error_dir)
        
        # Sample knowledge items
        self.entity_knowledge = {
            "id": "k_test1",
            "type": "entity_classification",
            "statement": "PAH is a gene name, not a disease abbreviation.",
            "entities": ["PAH"],
            "relations": [
                {"subject": "PAH", "predicate": "isA", "object": "gene"}
            ],
            "metadata": {
                "source": "error_feedback",
                "domain": "biomedical",
                "confidence": 0.8
            }
        }
        
        self.conceptual_knowledge = {
            "id": "k_test2",
            "type": "conceptual_knowledge",
            "statement": "HER2 is defined as a proto-oncogene located on chromosome 17q21.",
            "entities": ["HER2"],
            "relations": [
                {"subject": "HER2", "predicate": "isDefinedAs", "object": "proto-oncogene located on chromosome 17q21"}
            ],
            "metadata": {
                "source": "text_extraction",
                "domain": "biomedical",
                "confidence": 0.7
            }
        }
        
        self.procedural_knowledge = {
            "id": "k_test3",
            "type": "procedural_knowledge",
            "statement": "Procedure for sentiment analysis",
            "procedure_steps": [
                "Identify all subjective words and phrases",
                "Calculate the valence of each sentiment word",
                "Adjust for intensifiers and negations",
                "Combine individual scores for overall sentiment"
            ],
            "entities": ["sentiment analysis"],
            "metadata": {
                "source": "text_extraction",
                "domain": "nlp",
                "confidence": 0.75
            }
        }
    
    def teardown_method(self):
        """Clean up test resources."""
        self.temp_dir.cleanup()
    
    def test_add_retrieve_knowledge(self):
        """Test adding and retrieving knowledge."""
        # Add knowledge item
        self.kb.add_knowledge(self.entity_knowledge)
        
        # Retrieve domain knowledge
        domain_knowledge = self.kb.get_domain_knowledge("biomedical")
        
        # Check retrieval
        assert len(domain_knowledge) == 1
        assert domain_knowledge[0]["id"] == self.entity_knowledge["id"]
        
        # Get individual knowledge item
        item = self.kb.get_knowledge(self.entity_knowledge["id"])
        assert item is not None
        assert item["id"] == self.entity_knowledge["id"]
        
        # Add another item
        self.kb.add_knowledge(self.conceptual_knowledge)
        
        # Get updated domain knowledge
        updated_domain = self.kb.get_domain_knowledge("biomedical")
        assert len(updated_domain) == 2
    
    def test_add_knowledge_validation(self):
        """Test validation during knowledge addition."""
        # Missing required fields
        invalid_items = [
            {"statement": "Missing ID and type"},
            {"id": "k_missing_type", "statement": "Missing type"},
            {"id": "k_missing_statement", "type": "conceptual_knowledge"}
        ]
        
        # Try to add invalid items
        for item in invalid_items:
            with pytest.raises(ValueError):
                self.kb.add_knowledge(item)
        
        # Add valid item then try to add duplicate
        self.kb.add_knowledge(self.entity_knowledge)
        with pytest.raises(ValueError):
            self.kb.add_knowledge(self.entity_knowledge)  # Same ID
    
    def test_update_knowledge(self):
        """Test updating knowledge."""
        # Add knowledge item
        self.kb.add_knowledge(self.entity_knowledge)
        
        # Update the item
        updated_item = self.entity_knowledge.copy()
        updated_item["statement"] = "PAH is a gene involved in phenylalanine metabolism."
        
        # Perform update
        result = self.kb.update_knowledge(updated_item)
        assert result is True
        
        # Verify update
        item = self.kb.get_knowledge(self.entity_knowledge["id"])
        assert item["statement"] == updated_item["statement"]
        
        # Try to update non-existent item
        non_existent = {
            "id": "k_nonexistent",
            "type": "conceptual_knowledge",
            "statement": "This item doesn't exist"
        }
        result = self.kb.update_knowledge(non_existent)
        assert result is False
    
    def test_delete_knowledge(self):
        """Test deleting knowledge."""
        # Add knowledge items
        self.kb.add_knowledge(self.entity_knowledge)
        self.kb.add_knowledge(self.conceptual_knowledge)
        
        # Verify items exist
        domain_knowledge = self.kb.get_domain_knowledge("biomedical")
        assert len(domain_knowledge) == 2
        
        # Delete one item
        result = self.kb.delete_knowledge(self.entity_knowledge["id"])
        assert result is True
        
        # Verify deletion
        updated_domain = self.kb.get_domain_knowledge("biomedical")
        assert len(updated_domain) == 1
        assert updated_domain[0]["id"] == self.conceptual_knowledge["id"]
        
        # Try to delete non-existent item
        result = self.kb.delete_knowledge("k_nonexistent")
        assert result is False
    
    def test_search_knowledge(self):
        """Test knowledge search."""
        # Add multiple knowledge items
        self.kb.add_knowledge(self.entity_knowledge)
        self.kb.add_knowledge(self.conceptual_knowledge)
        self.kb.add_knowledge(self.procedural_knowledge)
        
        # Search by text
        pah_results = self.kb.search_knowledge("PAH gene")
        assert len(pah_results) == 1
        assert pah_results[0]["id"] == self.entity_knowledge["id"]
        
        # Search by domain
        biomedical_results = self.kb.search_knowledge("gene", domains=["biomedical"])
        assert len(biomedical_results) >= 1
        
        # Search by type
        entity_results = self.kb.search_knowledge("", types=["entity_classification"])
        assert len(entity_results) == 1
        assert entity_results[0]["type"] == "entity_classification"
        
        # Search with limit
        limited_results = self.kb.search_knowledge("", limit=1)
        assert len(limited_results) == 1
    
    def test_query_by_entities(self):
        """Test querying by entities."""
        # Add knowledge items
        self.kb.add_knowledge(self.entity_knowledge)
        self.kb.add_knowledge(self.conceptual_knowledge)
        
        # Query by entity
        pah_results = self.kb.query_by_entities(["PAH"])
        assert len(pah_results) == 1
        assert pah_results[0]["id"] == self.entity_knowledge["id"]
        
        # Query by multiple entities
        multi_results = self.kb.query_by_entities(["PAH", "HER2"])
        assert len(multi_results) == 2
        
        # Query by domain
        domain_results = self.kb.query_by_entities(["PAH"], domains=["biomedical"])
        assert len(domain_results) == 1
        
        # Query non-existent entity
        empty_results = self.kb.query_by_entities(["NONEXISTENT"])
        assert len(empty_results) == 0
    
    def test_get_domains(self):
        """Test getting domain information."""
        # Add knowledge to different domains
        self.kb.add_knowledge(self.entity_knowledge)  # biomedical
        self.kb.add_knowledge(self.procedural_knowledge)  # nlp
        
        # Get domains
        domains = self.kb.get_domains()
        
        # Should have two domains
        assert len(domains) == 2
        domain_names = [d["name"] for d in domains]
        assert "biomedical" in domain_names
        assert "nlp" in domain_names
        
        # Check domain stats
        for domain in domains:
            if domain["name"] == "biomedical":
                assert domain["count"] == 1
                assert "entity_classification" in domain["types"]
            elif domain["name"] == "nlp":
                assert domain["count"] == 1
                assert "procedural_knowledge" in domain["types"]
    
    def test_get_knowledge_stats(self):
        """Test getting knowledge statistics."""
        # Add knowledge items
        self.kb.add_knowledge(self.entity_knowledge)
        self.kb.add_knowledge(self.conceptual_knowledge)
        self.kb.add_knowledge(self.procedural_knowledge)
        
        # Get stats
        stats = self.kb.get_knowledge_stats()
        
        # Check stats
        assert stats["total_items"] == 3
        assert stats["domains"] == 2  # biomedical and nlp
        assert len(stats["types"]) == 3  # Three different types
        assert stats["types"]["entity_classification"] == 1
        assert stats["types"]["conceptual_knowledge"] == 1
        assert stats["types"]["procedural_knowledge"] == 1
    
    def test_error_patterns(self):
        """Test error pattern storage and retrieval."""
        # Create error pattern
        error_pattern = {
            "id": "p_test1",
            "pattern_type": "entity_confusion",
            "description": "PAH entity confusion pattern",
            "entities": ["PAH"],
            "frequency": 3,
            "examples": ["e1", "e2"]
        }
        
        # Add error pattern
        pattern_id = self.kb.add_error_pattern(error_pattern)
        assert pattern_id == error_pattern["id"]
        
        # Get error patterns
        patterns = self.kb.get_error_patterns("entity_confusion")
        assert len(patterns) == 1
        assert patterns[0]["id"] == error_pattern["id"]
        
        # Try invalid pattern
        invalid_pattern = {"description": "Missing ID and type"}
        with pytest.raises(ValueError):
            self.kb.add_error_pattern(invalid_pattern)
    
    def test_domain_knowledge_manager(self):
        """Test domain knowledge manager functionality."""
        # Initialize manager with knowledge base
        manager = DomainKnowledgeManager(self.kb)
        
        # Extract and add knowledge
        extracted = [self.entity_knowledge, self.conceptual_knowledge]
        
        # Add extracted knowledge
        ids = manager.add_knowledge(extracted, verify=True)
        assert len(ids) == 2
        
        # Query knowledge
        text_results = manager.query_knowledge(query="PAH gene")
        assert len(text_results) == 1
        
        entity_results = manager.query_knowledge(entities=["PAH"])
        assert len(entity_results) == 1
        
        # Get domain knowledge
        biomedical_knowledge = manager.get_domain_knowledge("biomedical")
        assert len(biomedical_knowledge) == 2