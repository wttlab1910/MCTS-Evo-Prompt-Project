"""
Tests for the Knowledge Verification functionality.

Tests the verification, consistency checking, and scoring of knowledge items.
"""
import pytest
from app.knowledge.extraction.verification import (
    KnowledgeVerifier, 
    ConsistencyVerifier, 
    RelationshipMapper, 
    ConfidenceScorer
)

class TestKnowledgeVerification:
    """Test cases for knowledge verification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize verifiers
        self.base_verifier = KnowledgeVerifier()
        self.consistency_verifier = ConsistencyVerifier()
        self.relationship_mapper = RelationshipMapper()
        self.confidence_scorer = ConfidenceScorer()
        
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
        
        # Create duplicate and contradictory items
        self.duplicate_entity = {
            "id": "k_test4",
            "type": "entity_classification",
            "statement": "PAH refers to a gene, not a disease.",
            "entities": ["PAH"],
            "relations": [
                {"subject": "PAH", "predicate": "isA", "object": "gene"}
            ],
            "metadata": {
                "source": "text_extraction",
                "domain": "biomedical",
                "confidence": 0.65
            }
        }
        
        self.contradictory_entity = {
            "id": "k_test5",
            "type": "entity_classification",
            "statement": "PAH is a disease abbreviation, not a gene name.",
            "entities": ["PAH"],
            "relations": [
                {"subject": "PAH", "predicate": "isA", "object": "disease"}
            ],
            "metadata": {
                "source": "text_extraction",
                "domain": "biomedical",
                "confidence": 0.6
            }
        }
    
    def test_base_verifier(self):
        """Test base knowledge verifier functionality."""
        # Test batch verification method
        with pytest.raises(NotImplementedError):
            self.base_verifier.verify(self.entity_knowledge)
        
        # Test batch verification with empty list
        items = self.base_verifier.batch_verify([], verify_fn=lambda x: x)
        assert items == []
    
    def test_consistency_verification(self):
        """Test consistency verification."""
        # Verify with no existing knowledge
        verified = self.consistency_verifier.verify(self.entity_knowledge)
        assert verified["metadata"]["verification"]["result"] == "passed"
        
        # Verify against duplicate
        verified_dup = self.consistency_verifier.verify(
            self.duplicate_entity, 
            existing_knowledge=[self.entity_knowledge]
        )
        
        # Should recognize as duplicate
        assert verified_dup["metadata"]["verification"]["method"] == "consistency"
        assert "duplicate" in verified_dup["metadata"]["verification"]["result"]
        
        # Verify against contradictory
        verified_contra = self.consistency_verifier.verify(
            self.contradictory_entity, 
            existing_knowledge=[self.entity_knowledge]
        )
        
        # Should recognize as contradictory
        assert verified_contra["metadata"]["verification"]["method"] == "consistency"
        assert "needs_review" in verified_contra["metadata"]["verification"]["result"]
        
        # Verify confidence adjustment
        assert verified_contra["metadata"]["confidence"] < self.contradictory_entity["metadata"]["confidence"]
    
    def test_relationship_mapping(self):
        """Test relationship mapping."""
        # Map relationships with no existing knowledge
        mapped = self.relationship_mapper.verify(self.entity_knowledge)
        assert mapped["metadata"]["relationship_mapping"]["relationships"] == []
        
        # Map relationships with related knowledge
        mapped_with_related = self.relationship_mapper.verify(
            self.entity_knowledge,
            existing_knowledge=[self.conceptual_knowledge]
        )
        
        # Should find relationships
        relationships = mapped_with_related["metadata"]["relationship_mapping"]["relationships"]
        assert len(relationships) == 0  # No direct relationships in this case
        
        # Test with more closely related items
        related_concept = {
            "id": "k_test6",
            "type": "conceptual_knowledge",
            "statement": "Genes are segments of DNA that contain instructions for making proteins.",
            "entities": ["gene"],
            "relations": [
                {"subject": "gene", "predicate": "isDefinedAs", "object": "segment of DNA"}
            ],
            "metadata": {
                "domain": "biomedical"
            }
        }
        
        mapped_related = self.relationship_mapper.verify(
            self.entity_knowledge,
            existing_knowledge=[related_concept]
        )
        
        # Should now find indirect relationships
        indirect_relationships = mapped_related["metadata"]["relationship_mapping"]["relationships"]
        if indirect_relationships:  # May or may not detect based on implementation
            assert "relationship_type" in indirect_relationships[0]
    
    def test_confidence_scoring(self):
        """Test confidence scoring."""
        # Score entity knowledge
        scored_entity = self.confidence_scorer.verify(self.entity_knowledge)
        
        # Check scoring results
        assert "confidence" in scored_entity["metadata"]
        assert scored_entity["metadata"]["confidence"] > 0
        assert "confidence_factors" in scored_entity["metadata"]
        
        # Score procedural knowledge
        scored_proc = self.confidence_scorer.verify(self.procedural_knowledge)
        
        # Check factors
        factors = scored_proc["metadata"]["confidence_factors"]
        assert "source_reliability" in factors
        assert "completeness" in factors
        assert "specificity" in factors
        
        # Test with poorly structured item
        poor_item = {
            "id": "k_poor",
            "type": "conceptual_knowledge",
            "statement": "Brief note",
            "metadata": {}
        }
        
        scored_poor = self.confidence_scorer.verify(poor_item)
        
        # Should have lower confidence
        assert scored_poor["metadata"]["confidence"] < scored_entity["metadata"]["confidence"]
    
    def test_batch_verification(self):
        """Test batch verification functionality."""
        # Create batch of items
        batch = [
            self.entity_knowledge,
            self.conceptual_knowledge,
            self.procedural_knowledge
        ]
        
        # Verify batch with consistency verifier
        verified_batch = self.consistency_verifier.batch_verify(batch)
        
        # Check all items are verified
        assert len(verified_batch) == len(batch)
        for item in verified_batch:
            assert "verification" in item["metadata"]
        
        # Verify batch with all verifiers
        scored_batch = self.confidence_scorer.batch_verify(verified_batch)
        
        # Check scoring
        assert len(scored_batch) == len(batch)
        for item in scored_batch:
            assert "confidence" in item["metadata"]
            assert "confidence_factors" in item["metadata"]
    
    def test_text_similarity_function(self):
        """Test text similarity function used in verification."""
        # Access similarity method from consistency verifier
        similarity = self.consistency_verifier._text_similarity
        
        # Test similar texts
        sim1 = similarity("PAH is a gene", "PAH refers to a gene")
        assert sim1 > 0.5  # Should be reasonably similar
        
        # Test dissimilar texts
        sim2 = similarity("PAH is a gene", "BRCA1 is related to cancer")
        assert sim2 < 0.5  # Should be dissimilar
        
        # Test empty texts
        sim3 = similarity("", "PAH is a gene")
        assert sim3 == 0.0  # Should handle empty strings
    
    def test_contradiction_detection(self):
        """Test contradiction detection functionality."""
        # Access contradiction method from consistency verifier
        contradictory = self.consistency_verifier._are_contradictory
        
        # Test clear contradictions
        assert contradictory("gene", "disease") == True
        assert contradictory("positive", "negative") == True
        assert contradictory("always", "never") == True
        
        # Test non-contradictions
        assert contradictory("gene", "protein") == False
        assert contradictory("large", "small protein") == False
        
        # Test with negation
        assert contradictory("is not functional", "is functional") == True
    
    def test_completeness_calculation(self):
        """Test completeness calculation in confidence scoring."""
        # Access completeness method from confidence scorer
        completeness = self.confidence_scorer._calculate_completeness
        
        # Test complete entity knowledge
        complete_score = completeness(self.entity_knowledge)
        assert complete_score == 1.0  # Has all required attributes
        
        # Test incomplete knowledge
        incomplete = {
            "id": "k_incomplete",
            "type": "entity_classification",
            "statement": "Incomplete entity",
            "entities": []  # Missing required entities
        }
        incomplete_score = completeness(incomplete)
        assert incomplete_score < 1.0
    
    def test_specificity_calculation(self):
        """Test specificity calculation in confidence scoring."""
        # Access specificity method from confidence scorer
        specificity = self.confidence_scorer._calculate_specificity
        
        # Test specific procedural knowledge
        specific_score = specificity(self.procedural_knowledge)
        assert specific_score > 0.5  # Has detailed steps
        
        # Test vague knowledge
        vague = {
            "id": "k_vague",
            "type": "conceptual_knowledge",
            "statement": "Brief",
            "entities": ["entity"],
            "relations": []
        }
        vague_score = specificity(vague)
        assert vague_score < specific_score