"""
Tests for the MDP state representation.
"""
import pytest
from app.core.mdp.state import PromptState

class TestPromptState:
    """
    Tests for PromptState class.
    """
    
    def test_state_initialization(self):
        """Test basic state initialization."""
        # Initialize with just text
        state = PromptState("This is a test prompt.")
        assert state.text == "This is a test prompt."
        assert isinstance(state.components, dict)
        assert isinstance(state.metrics, dict)
        assert isinstance(state.history, list)
        assert state.parent is None
        
        # Initialize with components
        components = {"role": "Test Role", "task": "Test Task"}
        state = PromptState("This is a test prompt.", components=components)
        assert state.components == components
        
        # Initialize with metrics
        metrics = {"performance": 0.7, "efficiency": 0.8}
        state = PromptState("This is a test prompt.", metrics=metrics)
        assert state.metrics == metrics
    
    def test_component_extraction(self):
        """Test component extraction from text."""
        text = """
        Role: Test Expert
        Task: Analyze the following content.
        
        Steps:
        - Read carefully
        - Identify key elements
        - Provide analysis
        
        Content: This is test content.
        Output Format: Provide analysis in bullet points.
        """
        
        state = PromptState(text)
        
        assert "role" in state.components
        assert "task" in state.components
        assert "steps" in state.components
        assert "content" in state.components
        assert "output_format" in state.components
        
        assert "Test Expert" in state.components["role"]
        assert "Analyze" in state.components["task"]
        assert len(state.components["steps"]) == 3
        assert "Read carefully" in state.components["steps"][0]
        assert "This is test content" in state.components["content"]
        assert "bullet points" in state.components["output_format"]
    
    def test_state_id_generation(self):
        """Test unique ID generation for states."""
        state1 = PromptState("This is a test prompt.")
        state2 = PromptState("This is a test prompt.")
        state3 = PromptState("This is a different prompt.")
        
        # Same text should yield same ID
        assert state1.state_id == state2.state_id
        
        # Different text should yield different ID
        assert state1.state_id != state3.state_id
    
    def test_structural_completeness(self):
        """Test structural completeness calculation."""
        # Empty state
        state = PromptState("")
        assert state.get_structural_completeness() == 0.0
        
        # Partial structure
        state = PromptState("Role: Test Role")
        assert state.get_structural_completeness() > 0.0
        assert state.get_structural_completeness() < 1.0
        
        # Complete structure
        text = """
        Role: Test Expert
        Task: Analyze the following content.
        
        Steps:
        - Read carefully
        - Identify key elements
        - Provide analysis
        
        Output Format: Provide analysis in bullet points.
        """
        state = PromptState(text)
        assert state.get_structural_completeness() == 1.0
    
    def test_state_equality(self):
        """Test state equality based on ID."""
        state1 = PromptState("This is a test prompt.")
        state2 = PromptState("This is a test prompt.")
        state3 = PromptState("This is a different prompt.")
        
        assert state1 == state2
        assert state1 != state3
        assert hash(state1) == hash(state2)
        assert hash(state1) != hash(state3)
    
    def test_state_copy(self):
        """Test state copying."""
        original = PromptState(
            "This is a test prompt.",
            components={"role": "Test Role"},
            metrics={"performance": 0.7}
        )
        
        copied = original.copy()
        
        # Same content
        assert copied.text == original.text
        assert copied.components == original.components
        assert copied.metrics == original.metrics
        
        # But different objects
        assert copied is not original
        assert copied.components is not original.components
        
        # Modify the copy should not affect the original
        copied.components["task"] = "New Task"
        assert "task" not in original.components