"""
Test knowledge integration functionality.
"""
import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.knowledge.extraction.extractor import ErrorBasedExtractor, ConceptualKnowledgeExtractor
from app.knowledge.extraction.verification import ConsistencyVerifier, ConfidenceScorer
from app.knowledge.integration.integrator import PromptKnowledgeIntegrator
from app.knowledge.integration.strategy import PlacementStrategy, FormatSelectionStrategy
from app.core.mdp.state import PromptState
from app.knowledge.knowledge_base import KnowledgeBase
from app.knowledge.domain.domain_knowledge import DomainKnowledgeManager

# Sample data for testing
@pytest.fixture
def sample_error_data():
    return [
        {
            "example_id": "e1",
            "example": {
                "text": "Patient has elevated PAH levels.",
                "expected": "gene_mention"
            },
            "error_type": "entity_confusion",
            "actual": "disease_mention",
            "description": "PAH was classified as a disease instead of a gene."
        },
        {
            "example_id": "e2",
            "example": {
                "text": "The study examined HBB mutations.",
                "expected": "gene_mention"
            },
            "error_type": "entity_confusion",
            "actual": "missed",
            "description": "HBB gene mention was missed entirely."
        }
    ]

@pytest.fixture
def sample_knowledge_item():
    return {
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

@pytest.fixture
def sample_prompt_state():
    text = """
    Role: Medical Text Analyzer
    Task: Analyze the provided medical text and identify gene mentions.
    
    Steps:
    - Read the medical text carefully
    - Identify all gene mentions
    - Exclude disease names and other biological entities
    - Return a list of identified genes
    
    Output Format: Provide a list of genes in the format "Gene: [gene_name]" followed by location in text.
    """
    return PromptState(text)

class TestKnowledgeExtraction:
    """Test knowledge extraction capabilities."""
    
    def test_error_based_extraction(self, sample_error_data):
        """Test extraction of knowledge from errors."""
        extractor = ErrorBasedExtractor()
        
        # Extract knowledge from errors
        knowledge = extractor.extract(sample_error_data, domain="biomedical")
        
        # Verify extracted knowledge
        assert len(knowledge) > 0
        
        # Check first knowledge item
        item = knowledge[0]
        assert item["type"] == "entity_classification"
        assert "PAH" in item["entities"]
        assert any("gene" in str(rel) for rel in item["relations"])
        assert "biomedical" in item["metadata"]["domain"]
    
    def test_conceptual_extraction(self):
        """Test extraction of conceptual knowledge."""
        extractor = ConceptualKnowledgeExtractor()
        
        # Sample text with conceptual knowledge
        text = """
        The HER2 gene is defined as a proto-oncogene located on chromosome 17q21.
        It encodes a member of the epidermal growth factor receptor family of receptor tyrosine kinases.
        """
        
        # Extract knowledge
        knowledge = extractor.extract(text, domain="biomedical")
        
        # Verify extracted knowledge
        assert len(knowledge) > 0
        
        # Check knowledge content
        item = knowledge[0]
        assert item["type"] == "conceptual_knowledge"
        assert "HER2" in item["entities"]
        assert any("isDefinedAs" in str(rel) for rel in item["relations"])
        assert "biomedical" in item["metadata"]["domain"]

class TestKnowledgeVerification:
    """Test knowledge verification capabilities."""
    
    def test_consistency_verification(self, sample_knowledge_item):
        """Test consistency verification."""
        verifier = ConsistencyVerifier()
        
        # Create a duplicate item with slightly different text
        duplicate = sample_knowledge_item.copy()
        duplicate["id"] = "k_test2"
        duplicate["statement"] = "PAH refers to a gene, not a disease."
        
        # Verify against existing knowledge
        verified = verifier.verify(
            duplicate, 
            existing_knowledge=[sample_knowledge_item]
        )
        
        # Check verification results
        assert verified["metadata"]["verification"]["method"] == "consistency"
        assert "duplicate" in verified["metadata"]["verification"]["result"]
        
        # Create a contradictory item
        contradiction = sample_knowledge_item.copy()
        contradiction["id"] = "k_test3"
        contradiction["statement"] = "PAH is a disease abbreviation, not a gene name."
        contradiction["relations"] = [
            {"subject": "PAH", "predicate": "isA", "object": "disease"}
        ]
        
        # Verify contradictory item
        verified = verifier.verify(
            contradiction, 
            existing_knowledge=[sample_knowledge_item]
        )
        
        # Check verification results
        assert verified["metadata"]["verification"]["method"] == "consistency"
        assert "needs_review" in verified["metadata"]["verification"]["result"]
    
    def test_confidence_scoring(self, sample_knowledge_item):
        """Test confidence scoring."""
        scorer = ConfidenceScorer()
        
        # Score a knowledge item
        scored = scorer.verify(sample_knowledge_item)
        
        # Check scoring results
        assert "confidence" in scored["metadata"]
        assert scored["metadata"]["confidence"] > 0
        assert "confidence_factors" in scored["metadata"]
        
        # Test with poorly structured item
        poor_item = {
            "id": "k_poor",
            "type": "conceptual_knowledge",
            "statement": "Brief note",
            "metadata": {}
        }
        
        scored_poor = scorer.verify(poor_item)
        
        # Check that confidence is lower
        assert scored_poor["metadata"]["confidence"] < scored["metadata"]["confidence"]

class TestKnowledgeIntegration:
    """Test knowledge integration capabilities."""
    
    def test_format_knowledge(self, sample_knowledge_item):
        """Test knowledge formatting."""
        integrator = PromptKnowledgeIntegrator()
        
        # Test different format types
        default_format = integrator.format_knowledge(sample_knowledge_item, "default")
        brief_format = integrator.format_knowledge(sample_knowledge_item, "brief")
        detailed_format = integrator.format_knowledge(sample_knowledge_item, "detailed")
        contrastive_format = integrator.format_knowledge(sample_knowledge_item, "contrastive")
        rule_format = integrator.format_knowledge(sample_knowledge_item, "rule")
        
        # Check formats
        assert "PAH" in default_format
        assert "gene" in default_format
        assert len(brief_format) < len(detailed_format)
        assert "PAH" in contrastive_format and "gene" in contrastive_format
        assert "Rule:" in rule_format
    
    def test_prompt_integration(self, sample_knowledge_item, sample_prompt_state):
        """Test integration into prompt state."""
        integrator = PromptKnowledgeIntegrator()
        
        # Integrate knowledge into prompt
        new_state = integrator.integrate(sample_prompt_state, sample_knowledge_item)
        
        # Check integration results
        assert "PAH" in new_state.text
        assert "gene" in new_state.text
        assert new_state.text != sample_prompt_state.text
        
        # Test with specific placement
        new_state2 = integrator.integrate(
            sample_prompt_state, 
            sample_knowledge_item,
            override_placement="constraints"
        )
        
        # Check constraint placement
        assert "Constraints:" in new_state2.text
        assert "PAH" in new_state2.text
    
    def test_placement_strategy(self, sample_knowledge_item, sample_prompt_state):
        """Test placement strategy."""
        strategy = PlacementStrategy()
        
        # Get placement for different knowledge types
        entity_placement = strategy.select_placement(sample_knowledge_item, sample_prompt_state)
        
        # Create a format knowledge item
        format_item = {
            "id": "k_format",
            "type": "format_specification",
            "statement": "List each gene on a separate line.",
            "format_rules": ["One gene per line", "Include position information"],
            "metadata": {"domain": "biomedical"}
        }
        
        format_placement = strategy.select_placement(format_item, sample_prompt_state)
        
        # Check placements
        assert entity_placement in strategy.placement_options
        assert format_placement == "format_instructions"  # Should target output format
    
    def test_format_selection(self, sample_knowledge_item, sample_prompt_state):
        """Test format selection strategy."""
        strategy = FormatSelectionStrategy()
        
        # Select formats for different placements
        knowledge_format = strategy.select_format(
            sample_knowledge_item, 
            sample_prompt_state, 
            "knowledge_section"
        )
        
        role_format = strategy.select_format(
            sample_knowledge_item, 
            sample_prompt_state, 
            "role_description"
        )
        
        steps_format = strategy.select_format(
            sample_knowledge_item, 
            sample_prompt_state, 
            "step_instructions"
        )
        
        # Check format selections
        assert knowledge_format in strategy.format_options
        assert role_format == "brief"  # Role descriptions should be brief
        assert steps_format == "rule"  # Steps should be rule-formatted

class TestKnowledgeBase:
    """Test knowledge base functionality."""
    
    @pytest.fixture
    def temp_kb_dir(self, tmp_path):
        """Create temporary knowledge base directory."""
        domain_dir = tmp_path / "domain_knowledge"
        error_dir = tmp_path / "error_patterns"
        
        domain_dir.mkdir()
        error_dir.mkdir()
        
        return domain_dir, error_dir
    
    def test_add_retrieve_knowledge(self, sample_knowledge_item, temp_kb_dir):
        """Test adding and retrieving knowledge."""
        domain_dir, error_dir = temp_kb_dir
        kb = KnowledgeBase(domain_dir, error_dir)
        
        # Add knowledge item
        kb.add_knowledge(sample_knowledge_item)
        
        # Retrieve domain knowledge
        domain_knowledge = kb.get_domain_knowledge("biomedical")
        
        # Check retrieval
        assert len(domain_knowledge) == 1
        assert domain_knowledge[0]["id"] == sample_knowledge_item["id"]
        
        # Get individual knowledge item
        item = kb.get_knowledge(sample_knowledge_item["id"])
        assert item is not None
        assert item["id"] == sample_knowledge_item["id"]
    
    def test_search_knowledge(self, sample_knowledge_item, temp_kb_dir):
        """Test knowledge search."""
        domain_dir, error_dir = temp_kb_dir
        kb = KnowledgeBase(domain_dir, error_dir)
        
        # Add multiple knowledge items
        kb.add_knowledge(sample_knowledge_item)
        
        # Create a second item
        item2 = sample_knowledge_item.copy()
        item2["id"] = "k_test2"
        item2["statement"] = "HER2 is an oncogene associated with breast cancer."
        item2["entities"] = ["HER2"]
        kb.add_knowledge(item2)
        
        # Search by text
        results1 = kb.search_knowledge("PAH gene")
        assert len(results1) == 1
        assert results1[0]["id"] == sample_knowledge_item["id"]
        
        # Search by entity
        results2 = kb.query_by_entities(["HER2"])
        assert len(results2) == 1
        assert results2[0]["id"] == "k_test2"
        
        # Search all entities
        results3 = kb.query_by_entities(["PAH", "HER2"])
        assert len(results3) == 2
    
    def test_update_delete_knowledge(self, sample_knowledge_item, temp_kb_dir):
        """Test updating and deleting knowledge."""
        domain_dir, error_dir = temp_kb_dir
        kb = KnowledgeBase(domain_dir, error_dir)
        
        # Add knowledge item
        kb.add_knowledge(sample_knowledge_item)
        
        # Update item
        updated_item = sample_knowledge_item.copy()
        updated_item["statement"] = "PAH is a gene involved in phenylalanine metabolism."
        kb.update_knowledge(updated_item)
        
        # Verify update
        item = kb.get_knowledge(sample_knowledge_item["id"])
        assert item["statement"] == updated_item["statement"]
        
        # Delete item
        result = kb.delete_knowledge(sample_knowledge_item["id"])
        assert result is True
        
        # Verify deletion
        domain_knowledge = kb.get_domain_knowledge("biomedical")
        assert len(domain_knowledge) == 0

class TestDomainKnowledgeManager:
    """Test domain knowledge manager functionality."""
    
    @pytest.fixture
    def mock_kb(self):
        """Create mock knowledge base."""
        kb = MagicMock(spec=KnowledgeBase)
        kb.get_domain_knowledge.return_value = []
        kb.add_knowledge.return_value = "k_test1"
        return kb
    
    def test_extract_verify_add(self, mock_kb, sample_error_data):
        """Test extraction, verification, and addition workflow."""
        # Create manager with mock knowledge base
        manager = DomainKnowledgeManager(mock_kb)
        
        # Extract knowledge
        extracted = manager.extract_knowledge(
            sample_error_data, 
            extractor_type="error",
            domain="biomedical"
        )
        
        assert len(extracted) > 0
        
        # Verify knowledge
        verified = manager.verify_knowledge(
            extracted,
            verify_types=["consistency", "confidence"]
        )
        
        assert len(verified) == len(extracted)
        
        # Add knowledge to knowledge base
        ids = manager.add_knowledge(verified, verify=False)
        
        assert len(ids) == len(verified)
        mock_kb.add_knowledge.assert_called()
    
    def test_query_knowledge(self, mock_kb):
        """Test knowledge querying."""
        # Create manager with mock knowledge base
        manager = DomainKnowledgeManager(mock_kb)
        
        # Setup mock responses
        mock_kb.search_knowledge.return_value = [{"id": "k1", "statement": "Test result"}]
        mock_kb.query_by_entities.return_value = [{"id": "k2", "statement": "Entity result"}]
        
        # Query by text
        results1 = manager.query_knowledge(query="test query", domains=["biomedical"])
        assert len(results1) == 1
        mock_kb.search_knowledge.assert_called_with(
            query="test query",
            domains=["biomedical"],
            types=None,
            limit=10
        )
        
        # Query by entities
        results2 = manager.query_knowledge(entities=["HER2"], domains=["biomedical"])
        assert len(results2) == 1
        mock_kb.query_by_entities.assert_called_with(
            entities=["HER2"],
            domains=["biomedical"],
            limit=10
        )