"""
Integration tests for the Phase 4 Knowledge System.

Tests the end-to-end functionality of the knowledge extraction, verification,
storage, and integration components working together.
"""
import pytest
import tempfile
from pathlib import Path
from app.core.mdp.state import PromptState
from app.knowledge.knowledge_base import KnowledgeBase
from app.knowledge.domain.domain_knowledge import DomainKnowledgeManager
from app.knowledge.extraction.extractor import ErrorBasedExtractor, ConceptualKnowledgeExtractor
from app.knowledge.extraction.verification import ConsistencyVerifier, ConfidenceScorer
from app.knowledge.integration.integrator import PromptKnowledgeIntegrator
from app.knowledge.error.error_analyzer import ErrorAnalyzer
from app.knowledge.error.feedback_generator import FeedbackGenerator

class TestPhase4KnowledgeSystem:
    """Integration tests for Phase 4 Knowledge System components."""
    
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
        
        # Initialize domain knowledge manager
        self.knowledge_manager = DomainKnowledgeManager(self.kb)
        
        # Initialize knowledge integrator
        self.integrator = PromptKnowledgeIntegrator()
        
        # Initialize extractors
        self.error_extractor = ErrorBasedExtractor()
        self.conceptual_extractor = ConceptualKnowledgeExtractor()
        
        # Initialize verifiers
        self.consistency_verifier = ConsistencyVerifier(self.kb)
        self.confidence_scorer = ConfidenceScorer()
        
        # Initialize error components
        self.error_analyzer = ErrorAnalyzer()
        self.feedback_generator = FeedbackGenerator()
        
        # Sample error data
        self.error_data = [
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
        
        # Sample text for conceptual knowledge
        self.conceptual_text = """
        The HER2 gene is defined as a proto-oncogene located on chromosome 17q21.
        It encodes a member of the epidermal growth factor receptor family of receptor tyrosine kinases.
        """
        
        # Sample prompt states
        self.basic_prompt = PromptState("Task: Identify genes in biomedical text.")
        
        self.structured_prompt = PromptState("""
        Role: Biomedical Analyst
        Task: Identify gene mentions in the biomedical text.
        
        Steps:
        - Read the biomedical text carefully
        - Identify all gene names and symbols
        - Mark their positions in the text
        
        Output Format: List of genes with their positions in the text.
        """)
    
    def teardown_method(self):
        """Clean up test resources."""
        self.temp_dir.cleanup()
    
    def test_end_to_end_error_based_workflow(self):
        """Test end-to-end workflow from errors to knowledge integration."""
        # Step 1: Analyze errors
        error_analysis = self.error_analyzer.analyze_errors(self.error_data)
        assert "patterns" in error_analysis
        
        # Step 2: Extract knowledge from errors
        knowledge_items = self.error_extractor.extract(
            self.error_data, 
            domain="biomedical",
            analysis=error_analysis
        )
        assert len(knowledge_items) > 0
        
        # Step 3: Verify knowledge
        verified_items = []
        for item in knowledge_items:
            verified = self.consistency_verifier.verify(item)
            scored = self.confidence_scorer.verify(verified)
            verified_items.append(scored)
        
        # Step 4: Store knowledge in KB
        added_ids = []
        for item in verified_items:
            try:
                item_id = self.kb.add_knowledge(item)
                added_ids.append(item_id)
            except ValueError:
                # Skip duplicates
                pass
        assert len(added_ids) > 0
        
        # Step 5: Generate feedback from error analysis
        feedback = self.feedback_generator.generate_feedback(error_analysis)
        assert len(feedback) > 0
        
        # Step 6: Integrate knowledge into prompt
        initial_prompt = self.basic_prompt
        final_prompt = initial_prompt
        
        for item_id in added_ids:
            knowledge = self.kb.get_knowledge(item_id)
            if knowledge:
                final_prompt = self.integrator.integrate(final_prompt, knowledge)
        
        # Verify knowledge was integrated
        assert "PAH" in final_prompt.text or "gene" in final_prompt.text
        assert len(final_prompt.text) > len(initial_prompt.text)
    
    def test_end_to_end_conceptual_workflow(self):
        """Test end-to-end workflow from text to knowledge integration."""
        # Step 1: Extract conceptual knowledge
        knowledge_items = self.conceptual_extractor.extract(
            self.conceptual_text,
            domain="biomedical"
        )
        assert len(knowledge_items) > 0
        
        # Check specific HER2 extraction
        her2_item = None
        for item in knowledge_items:
            if "HER2" in str(item.get("entities", [])):
                her2_item = item
                break
        
        assert her2_item is not None
        assert "HER2" in her2_item["entities"]
        
        # Step 2: Verify knowledge
        verified_items = []
        for item in knowledge_items:
            verified = self.consistency_verifier.verify(item)
            scored = self.confidence_scorer.verify(verified)
            verified_items.append(scored)
        
        # Step 3: Store knowledge in KB
        added_ids = []
        for item in verified_items:
            try:
                item_id = self.kb.add_knowledge(item)
                added_ids.append(item_id)
            except ValueError:
                # Skip duplicates
                pass
        assert len(added_ids) > 0
        
        # Step 4: Integrate knowledge into prompt
        initial_prompt = self.structured_prompt
        final_prompt = initial_prompt
        
        for item_id in added_ids:
            knowledge = self.kb.get_knowledge(item_id)
            if knowledge:
                final_prompt = self.integrator.integrate(final_prompt, knowledge)
        
        # Verify knowledge was integrated
        assert "HER2" in final_prompt.text
        assert "proto-oncogene" in final_prompt.text
        assert len(final_prompt.text) > len(initial_prompt.text)
    
    def test_domain_knowledge_manager_workflow(self):
        """Test end-to-end workflow using domain knowledge manager."""
        # Step 1: Extract knowledge
        error_knowledge = self.knowledge_manager.extract_knowledge(
            self.error_data,
            extractor_type="error",
            domain="biomedical"
        )
        assert len(error_knowledge) > 0
        
        conceptual_knowledge = self.knowledge_manager.extract_knowledge(
            self.conceptual_text,
            extractor_type="conceptual",
            domain="biomedical"
        )
        assert len(conceptual_knowledge) > 0
        
        # Combine knowledge
        all_knowledge = error_knowledge + conceptual_knowledge
        
        # Step 2: Verify knowledge and add to KB
        verified_knowledge = self.knowledge_manager.verify_knowledge(
            all_knowledge,
            verify_types=["consistency", "confidence"]
        )
        
        added_ids = self.knowledge_manager.add_knowledge(
            verified_knowledge,
            verify=False,  # Already verified
            extract_relationships=True
        )
        assert len(added_ids) > 0
        
        # Step 3: Query knowledge
        query_results = self.knowledge_manager.query_knowledge(
            query="gene",
            domains=["biomedical"]
        )
        assert len(query_results) > 0
        
        entity_results = self.knowledge_manager.query_knowledge(
            entities=["PAH", "HER2"],
            domains=["biomedical"]
        )
        assert len(entity_results) > 0
        
        # Step 4: Integrate knowledge into prompt
        initial_prompt = self.basic_prompt
        final_prompt = initial_prompt
        
        for knowledge in query_results[:2]:  # Use top 2 results
            final_prompt = self.integrator.integrate(final_prompt, knowledge)
        
        # Verify knowledge was integrated
        assert len(final_prompt.text) > len(initial_prompt.text)
        assert "gene" in final_prompt.text
        assert "PAH" in final_prompt.text or "HER2" in final_prompt.text
    
    def test_knowledge_integration_strategies(self):
        """Test different knowledge integration strategies."""
        # Extract and store knowledge
        knowledge_items = self.conceptual_extractor.extract(
            self.conceptual_text,
            domain="biomedical"
        )
        
        # Find HER2 knowledge
        her2_knowledge = None
        for item in knowledge_items:
            if "HER2" in str(item.get("entities", [])):
                her2_knowledge = item
                break
        
        assert her2_knowledge is not None
        
        # Test different placement strategies
        placements = [
            "knowledge_section",
            "role_description",
            "task_description",
            "step_instructions",
            "constraints"
        ]
        
        for placement in placements:
            # Integrate with specific placement
            placed_prompt = self.integrator.integrate(
                self.structured_prompt,
                her2_knowledge,
                override_placement=placement
            )
            
            # Verify placement
            assert "HER2" in placed_prompt.text
            
            # Different placements should yield different results
            if placement == "role_description":
                assert "Role:" in placed_prompt.text
                assert "HER2" in placed_prompt.text.split("\n\n")[0]  # Should be in first section
            elif placement == "knowledge_section":
                assert "Domain Knowledge:" in placed_prompt.text or "Knowledge:" in placed_prompt.text
            elif placement == "constraints":
                assert "Constraints:" in placed_prompt.text
        
        # Test different format strategies
        formats = ["brief", "detailed", "rule"]
        
        for format_type in formats:
            # Integrate with specific format
            formatted_prompt = self.integrator.integrate(
                self.structured_prompt,
                her2_knowledge,
                override_format=format_type
            )
            
            # Verify format
            assert "HER2" in formatted_prompt.text
            
            # Different formats should yield different results
            if format_type == "brief":
                assert len(formatted_prompt.text) < len(self.integrator.integrate(
                    self.structured_prompt, her2_knowledge, override_format="detailed"
                ).text)
            elif format_type == "rule":
                assert "Rule:" in formatted_prompt.text