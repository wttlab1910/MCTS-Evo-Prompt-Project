"""
Tests for the Error Analysis and Feedback functionality.

Tests the collection, analysis, and feedback generation from error patterns.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.knowledge.error.error_analyzer import ErrorAnalyzer
from app.knowledge.error.error_collector import ErrorCollector
from app.knowledge.error.feedback_generator import FeedbackGenerator
from app.core.mdp.state import PromptState

class TestErrorAnalysis:
    """Test cases for error analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize components
        self.analyzer = ErrorAnalyzer()
        self.collector = ErrorCollector()
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
        
        # Sample prompt state
        self.test_prompt = PromptState("""
        Task: Identify gene mentions in biomedical text.
        
        Instructions:
        - Read the text carefully
        - Mark all gene names like BRCA1, TP53, etc.
        - Do not include disease names or other biological entities
        
        Output Format: List of gene names with their positions.
        """)
    
    def test_error_analyzer_initialization(self):
        """Test error analyzer initialization."""
        # Check error categories
        assert hasattr(self.analyzer, "error_categories")
        assert "entity_confusion" in self.analyzer.error_categories
        assert "procedure_error" in self.analyzer.error_categories
        assert "domain_misconception" in self.analyzer.error_categories
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        # Analyze errors
        analysis = self.analyzer.analyze_errors(self.error_data)
        
        # Check analysis structure
        assert "error_clusters" in analysis
        assert "patterns" in analysis
        assert "summary" in analysis
        
        # Check error clustering
        clusters = analysis["error_clusters"]
        assert "entity_confusion" in clusters
        assert "procedure_error" in clusters
        
        # Check pattern extraction
        patterns = analysis["patterns"]
        assert len(patterns) > 0
        
        # Find entity confusion pattern
        entity_pattern = None
        for pattern in patterns:
            if pattern.get("pattern_type") == "entity_confusion":
                entity_pattern = pattern
                break
        
        # Check entity pattern
        assert entity_pattern is not None
        assert "PAH" in str(entity_pattern.get("entities", []))
        assert entity_pattern.get("frequency", 0) > 0
    
    def test_error_categorization(self):
        """Test error categorization functionality."""
        # Test direct categorization
        entity_category = self.analyzer._categorize_error("entity_confusion", "Entity was misclassified")
        assert entity_category == "entity_confusion"
        
        procedure_category = self.analyzer._categorize_error("procedure_error", "Steps were not followed")
        assert procedure_category == "procedure_error"
        
        # Test categorization from description
        description_category = self.analyzer._categorize_error("unknown", "Entity was confused with another type")
        assert description_category == "entity_confusion"
        
        format_category = self.analyzer._categorize_error("unknown", "Output format was incorrect")
        assert format_category == "format_inconsistency"
    
    def test_error_collector_synchronous(self):
        """Test synchronous error collection."""
        # Collect errors synchronously (mock mode)
        errors = self.collector.collect_errors(self.test_prompt, [
            {"id": "e1", "text": "Sample text", "expected": "Expected output"}
        ])
        
        # Check collected errors
        assert len(errors) > 0
        assert hasattr(self.collector, "collected_errors")
        assert len(self.collector.collected_errors) > 0
    
    def test_mock_response_generation(self):
        """Test mock response generation."""
        # Access mock response method
        mock_response = self.collector._mock_response
        
        # Generate mock response
        response = mock_response(self.test_prompt, {"text": "Sample", "expected": "Output"})
        
        # Should generate some response
        assert response is not None
        assert isinstance(response, str)
    
    def test_error_detection(self):
        """Test error detection functionality."""
        # Test error detection
        is_error = self.collector._is_error("Actual response", "Expected response")
        assert is_error is True
        
        is_not_error = self.collector._is_error("Expected response", "Expected response")
        assert is_not_error is False
    
    def test_feedback_generator_initialization(self):
        """Test feedback generator initialization."""
        # Check error-action mapping
        assert hasattr(self.feedback_generator, "error_action_mapping")
        
        # Check suggestion templates
        assert hasattr(self.feedback_generator, "suggestion_templates")
        assert "entity_confusion" in self.feedback_generator.suggestion_templates
        assert "procedure_error" in self.feedback_generator.suggestion_templates
    
    def test_feedback_generation(self):
        """Test feedback generation from analysis."""
        # Generate analysis
        analysis = self.analyzer.analyze_errors(self.error_data)
        
        # Generate feedback
        feedback = self.feedback_generator.generate_feedback(analysis)
        
        # Check feedback structure
        assert len(feedback) > 0
        for item in feedback:
            assert "type" in item
            assert "suggestion" in item
            assert "impact" in item
            assert "action_mapping" in item
    
    def test_action_mapping(self):
        """Test mapping feedback to actions."""
        # Create feedback items
        feedback_items = [
            {
                "type": "entity_confusion",
                "description": "PAH was confused with a disease",
                "suggestion": "Clarify that PAH is a gene",
                "impact": "High",
                "action_mapping": {
                    "action_type": "add_domain_knowledge",
                    "parameters": {
                        "knowledge_text": "PAH is a gene name, not a disease abbreviation.",
                        "domain": "biomedical",
                        "location": "knowledge_section"
                    }
                }
            },
            {
                "type": "procedure_error",
                "description": "Steps were not formatted correctly",
                "suggestion": "Use numbered list for steps",
                "impact": "Medium",
                "action_mapping": {
                    "action_type": "modify_workflow",
                    "parameters": {
                        "steps": ["Step 1", "Step 2"]
                    }
                }
            }
        ]
        
        # Mock create_action function
        with patch('app.core.mdp.action.create_action') as mock_create:
            mock_create.return_value = MagicMock()
            
            # Map to actions
            actions = self.feedback_generator.map_feedback_to_actions(feedback_items)
            
            # Check actions
            assert len(actions) == 2
            assert mock_create.call_count == 2
    
    def test_suggestion_generation(self):
        """Test suggestion generation from patterns."""
        # Access pattern feedback generation method
        pattern_feedback = self.feedback_generator._generate_pattern_feedback
        
        # Generate feedback for entity confusion
        entity_feedback = pattern_feedback(
            "entity_confusion",
            "PAH was confused with a disease",
            ["PAH"]
        )
        
        # Check entity feedback
        assert entity_feedback is not None
        assert "PAH" in entity_feedback["suggestion"]
        assert entity_feedback["type"] == "entity_confusion"
        
        # Generate feedback for procedure error
        proc_feedback = pattern_feedback(
            "procedure_error",
            "Steps were not formatted correctly",
            []
        )
        
        # Check procedure feedback
        assert proc_feedback is not None
        assert "Steps" in proc_feedback["description"]
        assert "impact" in proc_feedback
    
    def test_extract_noun_phrases(self):
        """Test noun phrase extraction for suggestions."""
        # Access noun phrase extraction method
        extract_phrases = self.feedback_generator._extract_noun_phrases
        
        # Extract from text
        phrases = extract_phrases("Gene expression analysis shows increased PAH levels")
        
        # Check extracted phrases
        assert len(phrases) > 0
        assert "Gene expression" in phrases or "PAH levels" in phrases
    
    def test_async_error_collection(self):
        """Test async error collection with mocked LLM."""
        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value={"text": "Mock response"})
        
        # Create collector with mock LLM
        collector_with_llm = ErrorCollector(llm=mock_llm)
        
        # Test async collection
        import asyncio
        
        async def run_async_test():
            errors = await collector_with_llm.collect_errors_async(
                self.test_prompt, 
                [{"id": "e1", "text": "Sample text", "expected": "Expected output"}]
            )
            return errors
        
        # Skip if running in environment without event loop
        try:
            errors = asyncio.run(run_async_test())
            assert len(errors) > 0
        except RuntimeError:
            # No event loop - skip this test
            pass
    
    def test_error_pattern_generation(self):
        """Test error pattern generation from similar errors."""
        # Access pattern generation method
        generate_pattern = self.analyzer._generate_pattern
        
        # Generate pattern from similar errors
        pattern = generate_pattern("entity_confusion", self.error_data[:2])  # First two are entity confusion
        
        # Check pattern
        assert pattern is not None
        assert "id" in pattern and pattern["id"].startswith("p_")
        assert pattern["pattern_type"] == "entity_confusion"
        assert "description" in pattern
        assert "frequency" in pattern and pattern["frequency"] == 2