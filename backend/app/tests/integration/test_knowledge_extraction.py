"""
Tests for the Knowledge Extraction functionality.

Tests the extraction of knowledge from various sources including errors and text.
"""
import pytest
from app.knowledge.extraction.extractor import (
    KnowledgeExtractor, 
    ErrorBasedExtractor, 
    ConceptualKnowledgeExtractor,
    ProceduralKnowledgeExtractor
)
from app.knowledge.error.error_analyzer import ErrorAnalyzer

class TestKnowledgeExtraction:
    """Test cases for knowledge extraction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize extractors
        self.base_extractor = KnowledgeExtractor()
        self.error_extractor = ErrorBasedExtractor()
        self.conceptual_extractor = ConceptualKnowledgeExtractor()
        self.procedural_extractor = ProceduralKnowledgeExtractor()
        
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
            },
            {
                "example_id": "e3",
                "example": {
                    "text": "The patient was treated according to protocol.",
                    "expected": "formatted_steps"
                },
                "error_type": "procedure_error",
                "actual": "unformatted_text",
                "description": "Treatment steps were not properly formatted as a numbered list."
            }
        ]
        
        # Sample text for conceptual knowledge extraction
        self.conceptual_text = """
        The HER2 gene is defined as a proto-oncogene located on chromosome 17q21.
        It encodes a member of the epidermal growth factor receptor family of receptor tyrosine kinases.
        
        KRAS is a gene that provides instructions for making the K-Ras protein.
        
        Cytokines are a broad category of small proteins that are important in cell signaling.
        """
        
        # Sample text for procedural knowledge extraction
        self.procedural_text = """
        How to perform sentiment analysis:
        
        1. Read the text carefully to understand context.
        2. Identify all subjective words and phrases (adjectives, adverbs).
        3. Assign polarity values to each subjective element (positive or negative).
        4. Account for negations and intensifiers that modify sentiment.
        5. Calculate overall sentiment score based on individual elements.
        6. Classify the text as positive, negative, or neutral based on the score.
        
        Remember to consider cultural and domain-specific context when interpreting sentiment.
        """
    
    def test_base_extractor(self):
        """Test base knowledge extractor functionality."""
        # Test formalize method
        knowledge = {
            "type": "test_knowledge",
            "statement": "This is a test statement",
            "entities": ["test"],
            "confidence": 0.7
        }
        
        formalized = self.base_extractor.formalize(knowledge)
        
        # Check formalized structure
        assert "id" in formalized and formalized["id"].startswith("k_")
        assert formalized["type"] == "test_knowledge"
        assert formalized["statement"] == "This is a test statement"
        assert formalized["entities"] == ["test"]
        assert "metadata" in formalized
        assert formalized["metadata"]["confidence"] == 0.7
    
    def test_error_based_extraction(self):
        """Test extraction of knowledge from errors."""
        # Extract knowledge from error data
        knowledge_items = self.error_extractor.extract(self.error_data, domain="biomedical")
        
        # Verify extraction results
        assert len(knowledge_items) > 0
        assert len(self.error_extractor.get_extracted_knowledge()) > 0
        
        # Find entity classification for PAH
        pah_knowledge = None
        for item in knowledge_items:
            if "PAH" in str(item.get("entities", [])):
                pah_knowledge = item
                break
        
        # Check PAH knowledge
        assert pah_knowledge is not None
        assert pah_knowledge["type"] == "entity_classification"
        assert "PAH" in pah_knowledge["entities"]
        assert any("gene" in str(rel) for rel in pah_knowledge["relations"])
        assert pah_knowledge["metadata"]["domain"] == "biomedical"
        
        # Check for procedural knowledge
        proc_knowledge = None
        for item in knowledge_items:
            if item["type"] == "procedural_knowledge":
                proc_knowledge = item
                break
        
        # Should extract procedural knowledge from procedure_error
        assert proc_knowledge is not None
        assert "steps" in str(proc_knowledge).lower() or "format" in str(proc_knowledge).lower()
    
    def test_conceptual_knowledge_extraction(self):
        """Test extraction of conceptual knowledge from text."""
        # Extract conceptual knowledge
        knowledge_items = self.conceptual_extractor.extract(self.conceptual_text, domain="biomedical")
        
        # Verify extraction results
        assert len(knowledge_items) > 0
        
        # Find HER2 knowledge
        her2_knowledge = None
        for item in knowledge_items:
            if "HER2" in str(item.get("entities", [])):
                her2_knowledge = item
                break
        
        # Check HER2 knowledge
        assert her2_knowledge is not None
        assert her2_knowledge["type"] == "conceptual_knowledge"
        assert "HER2" in her2_knowledge["entities"]
        assert any("isDefinedAs" in str(rel) for rel in her2_knowledge["relations"])
        assert "proto-oncogene" in str(her2_knowledge)
        
        # Find KRAS knowledge
        kras_knowledge = None
        for item in knowledge_items:
            if "KRAS" in str(item.get("entities", [])):
                kras_knowledge = item
                break
        
        # Check for multiple entity extraction
        if kras_knowledge:
            assert kras_knowledge["type"] == "conceptual_knowledge"
            assert "KRAS" in kras_knowledge["entities"]
    
    def test_procedural_knowledge_extraction(self):
        """Test extraction of procedural knowledge from text."""
        # Extract procedural knowledge
        knowledge_items = self.procedural_extractor.extract(self.procedural_text, domain="nlp")
        
        # Verify extraction results
        assert len(knowledge_items) > 0
        assert len(knowledge_items[0]["procedure_steps"]) >= 3  # Should extract multiple steps
        
        # Check extracted knowledge
        proc_knowledge = knowledge_items[0]
        assert proc_knowledge["type"] == "procedural_knowledge"
        assert "sentiment analysis" in str(proc_knowledge).lower()
        assert any("read the text" in step.lower() for step in proc_knowledge["procedure_steps"])
        assert proc_knowledge["metadata"]["domain"] == "nlp"
    
    def test_extraction_from_partial_information(self):
        """Test extraction with incomplete information."""
        # Create minimal error data
        minimal_error = [{
            "error_type": "entity_confusion",
            "description": "Confused entity X with Y"
        }]
        
        # Extract knowledge from minimal error
        knowledge_items = self.error_extractor.extract(minimal_error)
        
        # Should still extract something
        assert len(knowledge_items) > 0
        
        # Test with minimal conceptual text
        minimal_text = "X is defined as Y."
        conceptual_items = self.conceptual_extractor.extract(minimal_text)
        
        # May or may not extract (depends on implementation)
        if conceptual_items:
            assert conceptual_items[0]["type"] == "conceptual_knowledge"
    
    def test_extraction_with_custom_parameters(self):
        """Test extraction with custom parameters."""
        # Extract with custom domain
        custom_domain = "custom_domain"
        knowledge_items = self.error_extractor.extract(self.error_data, domain=custom_domain)
        
        # Check domain
        assert all(item["metadata"]["domain"] == custom_domain for item in knowledge_items)
        
        # Extract with task type
        task_type = "entity_recognition"
        task_items = self.error_extractor.extract(self.error_data, domain=custom_domain, task_type=task_type)
        
        # Result should be similar (task_type mainly affects internal processing)
        assert len(task_items) > 0
    
    def test_extraction_with_analyzer(self):
        """Test extraction with provided error analysis."""
        # Create analyzer and analyze errors
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze_errors(self.error_data)
        
        # Extract with analysis
        knowledge_items = self.error_extractor.extract(self.error_data, analysis=analysis)
        
        # Should extract patterns from analysis
        assert len(knowledge_items) > 0
        
        # Extract directly from errors (for comparison)
        direct_items = self.error_extractor.extract(self.error_data)
        
        # Results may differ but should be structurally similar
        assert len(knowledge_items) > 0 and len(direct_items) > 0
    
    def test_her2_gene_extraction(self):
        """Test specific HER2 gene extraction case."""
        # Create test case for HER2 gene extraction
        her2_text = "The HER2 gene is defined as a proto-oncogene located on chromosome 17q21."
        
        # Extract knowledge
        knowledge_items = self.conceptual_extractor.extract(her2_text, domain="biomedical")
        
        # Verify extraction results
        assert len(knowledge_items) > 0
        
        # Check HER2 extraction
        item = knowledge_items[0]
        assert item["type"] == "conceptual_knowledge"
        assert "HER2" in item["entities"]  # Important: should extract "HER2" not "gene"
        assert any("isDefinedAs" in str(rel) for rel in item["relations"])
        assert "proto-oncogene" in item["statement"]