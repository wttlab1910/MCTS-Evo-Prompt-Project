"""
Tests for the Knowledge Integration functionality.

Tests the integration of knowledge into prompts and the various integration strategies.
"""
import pytest
from app.core.mdp.state import PromptState
from app.knowledge.integration.integrator import KnowledgeIntegrator, PromptKnowledgeIntegrator
from app.knowledge.integration.strategy import (
    PlacementStrategy, 
    FormatSelectionStrategy, 
    ConflictResolutionStrategy,
    TemplateIntegrationStrategy
)

class TestKnowledgeIntegration:
    """Test cases for knowledge integration into prompts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test prompt states
        self.basic_prompt = PromptState("Task: Analyze the sentiment of the text.")
        
        self.structured_prompt = PromptState("""
        Role: Sentiment Analyst
        Task: Analyze the sentiment of the provided text.
        
        Steps:
        - Read the text carefully
        - Identify sentiment words and phrases
        - Determine overall sentiment
        
        Output Format: Provide sentiment as positive, negative, or neutral.
        """)
        
        # Create test knowledge items
        self.entity_knowledge = {
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
        
        self.procedural_knowledge = {
            "type": "procedural_knowledge",
            "statement": "Process for sentiment analysis",
            "procedure_steps": [
                "Identify all subjective words and phrases",
                "Calculate the valence of each sentiment word",
                "Adjust for intensifiers and negations",
                "Combine individual scores for overall sentiment"
            ],
            "entities": ["sentiment analysis"],
            "metadata": {
                "domain": "nlp",
                "confidence": 0.9
            }
        }
        
        self.format_knowledge = {
            "type": "format_specification",
            "statement": "Format specification for sentiment analysis",
            "format_rules": [
                "Include an overall sentiment label (positive, negative, neutral)",
                "Provide confidence score between 0 and 1",
                "List key sentiment words that influenced the decision"
            ],
            "entities": ["sentiment analysis", "format"],
            "metadata": {
                "domain": "nlp",
                "confidence": 0.85
            }
        }
        
        # Create integrators
        self.basic_integrator = PromptKnowledgeIntegrator()
        
        # Create custom integrators with specific strategies
        self.placement_strategy = PlacementStrategy()
        self.format_strategy = FormatSelectionStrategy()
        self.conflict_strategy = ConflictResolutionStrategy()
        
        self.custom_integrator = PromptKnowledgeIntegrator(
            placement_strategy=self.placement_strategy,
            format_strategy=self.format_strategy,
            conflict_strategy=self.conflict_strategy
        )
    
    def test_integrator_initialization(self):
        """Test integrator initialization."""
        # Test base class
        base_integrator = KnowledgeIntegrator()
        assert hasattr(base_integrator, "format_knowledge")
        
        # Test prompt integrator with default strategies
        assert self.basic_integrator.placement_strategy is not None
        assert self.basic_integrator.format_strategy is not None
        assert self.basic_integrator.conflict_strategy is not None
        
        # Test custom integrator
        assert self.custom_integrator.placement_strategy == self.placement_strategy
        assert self.custom_integrator.format_strategy == self.format_strategy
        assert self.custom_integrator.conflict_strategy == self.conflict_strategy
    
    def test_knowledge_formatting(self):
        """Test knowledge formatting functionality."""
        # Test different format types
        default_format = self.basic_integrator.format_knowledge(self.entity_knowledge, "default")
        brief_format = self.basic_integrator.format_knowledge(self.entity_knowledge, "brief")
        detailed_format = self.basic_integrator.format_knowledge(self.entity_knowledge, "detailed")
        contrastive_format = self.basic_integrator.format_knowledge(self.entity_knowledge, "contrastive")
        rule_format = self.basic_integrator.format_knowledge(self.entity_knowledge, "rule")
        
        # Check formats
        assert "PAH" in default_format
        assert "gene" in default_format
        assert len(brief_format) < len(detailed_format)
        assert "PAH" in brief_format
        assert "entities" in detailed_format
        assert "gene" in contrastive_format
        assert "Rule:" in rule_format
        
        # Test procedural knowledge formatting
        proc_default = self.basic_integrator.format_knowledge(self.procedural_knowledge, "default")
        proc_detailed = self.basic_integrator.format_knowledge(self.procedural_knowledge, "detailed")
        
        # Check procedural formats
        assert "sentiment analysis" in proc_default
        assert "Steps:" in proc_detailed or "steps" in proc_detailed.lower()
    
    def test_integration_into_basic_prompt(self):
        """Test integration into a basic prompt."""
        # Integrate entity knowledge
        new_state = self.basic_integrator.integrate(self.basic_prompt, self.entity_knowledge)
        
        # Check integration results
        assert "PAH" in new_state.text
        assert "gene" in new_state.text
        assert new_state.text != self.basic_prompt.text
        
        # Verify parent/child relationship
        assert new_state.parent == self.basic_prompt
        assert "knowledge_integration" in new_state.history
    
    def test_integration_into_structured_prompt(self):
        """Test integration into a structured prompt."""
        # Integrate procedural knowledge
        new_state = self.basic_integrator.integrate(self.structured_prompt, self.procedural_knowledge)
        
        # Check integration results
        assert "sentiment analysis" in new_state.text
        assert "Identify all subjective words" in new_state.text
        assert new_state.text != self.structured_prompt.text
        
        # Integrate format knowledge
        format_state = self.basic_integrator.integrate(self.structured_prompt, self.format_knowledge)
        
        # Check format integration
        assert "Format" in format_state.text
        assert "confidence score" in format_state.text
        assert "sentiment words" in format_state.text
    
    def test_placement_strategy(self):
        """Test placement strategy functionality."""
        # Test different placements
        entity_placement = self.placement_strategy.select_placement(self.entity_knowledge, self.structured_prompt)
        proc_placement = self.placement_strategy.select_placement(self.procedural_knowledge, self.structured_prompt)
        format_placement = self.placement_strategy.select_placement(self.format_knowledge, self.structured_prompt)
        
        # Check placements
        assert entity_placement in self.placement_strategy.placement_options
        assert proc_placement in self.placement_strategy.placement_options
        assert format_placement == "format_instructions"  # Format should target output format
        
        # Test integrating with specific placements
        entity_state = self.basic_integrator.integrate(
            self.structured_prompt, 
            self.entity_knowledge,
            override_placement="role_description"
        )
        
        # Check specific placement results
        assert "Role:" in entity_state.text
        assert "PAH" in entity_state.text
    
    def test_format_selection_strategy(self):
        """Test format selection strategy."""
        # Test format selection for different placements
        entity_knowledge_format = self.format_strategy.select_format(
            self.entity_knowledge, 
            self.structured_prompt, 
            "knowledge_section"
        )
        
        role_format = self.format_strategy.select_format(
            self.entity_knowledge, 
            self.structured_prompt, 
            "role_description"
        )
        
        steps_format = self.format_strategy.select_format(
            self.procedural_knowledge, 
            self.structured_prompt, 
            "step_instructions"
        )
        
        # Check format selections
        assert entity_knowledge_format in self.format_strategy.format_options
        assert role_format == "brief"  # Role descriptions should be brief
        assert steps_format == "rule"  # Steps should be rule-formatted
        
        # Test integration with specific formats
        detailed_state = self.basic_integrator.integrate(
            self.structured_prompt, 
            self.entity_knowledge,
            override_format="detailed"
        )
        
        # Check formatting in integration
        assert "PAH" in detailed_state.text
        assert "entities" in detailed_state.text or "gene" in detailed_state.text
    
    def test_conflict_resolution(self):
        """Test conflict resolution strategy."""
        # Create conflicting knowledge
        conflicting_knowledge = {
            "type": "entity_classification",
            "statement": "PAH is a disease abbreviation, not a gene name.",
            "entities": ["PAH"],
            "relations": [
                {"subject": "PAH", "predicate": "isA", "object": "disease"}
            ],
            "metadata": {
                "source": "error_feedback",
                "domain": "biomedical",
                "confidence": 0.6
            }
        }
        
        # Create prompt with existing knowledge
        prompt_with_knowledge = PromptState("""
        Role: Biomedical Analyst
        Task: Analyze the biomedical entities in the text.
        
        Domain Knowledge:
        - PAH is a gene name, not a disease abbreviation.
        
        Output Format: List all genes and proteins.
        """)
        
        # Resolve conflicts
        new_knowledge = self.conflict_strategy.resolve_conflicts(
            [conflicting_knowledge], 
            ["PAH is a gene name, not a disease abbreviation."]
        )
        
        # Should detect the conflict and not include the conflicting knowledge
        assert len(new_knowledge) == 0 or new_knowledge[0].get("metadata", {}).get("verification", {}).get("result") != "passed"
        
        # Try to integrate conflicting knowledge
        conflict_state = self.basic_integrator.integrate(prompt_with_knowledge, conflicting_knowledge)
        
        # Should not duplicate or add contradictory information
        assert conflict_state.text.count("PAH is a gene") == 1
        assert "PAH is a disease" not in conflict_state.text
    
    def test_template_integration(self):
        """Test template-based integration."""
        # Create template integrator
        template_strategy = TemplateIntegrationStrategy()
        
        # Test template application
        template_text = template_strategy.apply_template(
            self.entity_knowledge,
            self.structured_prompt,
            "knowledge_section"
        )
        
        # Check template formatting
        assert "Domain Knowledge:" in template_text
        assert "Entity classification:" in template_text or "PAH" in template_text
        
        # Test template with procedural knowledge
        proc_template = template_strategy.apply_template(
            self.procedural_knowledge,
            self.structured_prompt,
            "step_instructions"
        )
        
        # Check procedural template
        assert "Steps:" in proc_template or "Important Concept:" in proc_template
    
    def test_multiple_knowledge_integration(self):
        """Test integrating multiple knowledge items."""
        # Create list of knowledge items
        knowledge_items = [
            self.entity_knowledge,
            self.procedural_knowledge,
            self.format_knowledge
        ]
        
        # Integrate multiple items
        new_state = self.basic_prompt
        for item in knowledge_items:
            new_state = self.basic_integrator.integrate(new_state, item)
        
        # Check comprehensive integration
        assert "PAH" in new_state.text
        assert "gene" in new_state.text
        assert "sentiment analysis" in new_state.text
        assert "sentiment words" in new_state.text
        
        # Check section organization
        sections = new_state.text.split("\n\n")
        assert len(sections) > 1  # Should have multiple sections