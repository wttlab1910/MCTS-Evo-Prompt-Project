"""
Tests for the MDP state transitions.
"""
import pytest
from app.core.mdp.state import PromptState
from app.core.mdp.action import AddRoleAction, ModifyWorkflowAction
from app.core.mdp.transition import StateTransition

class TestStateTransition:
    """
    Tests for state transitions.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.empty_state = PromptState("")
        
        # Create a basic state
        self.basic_state = PromptState(
            """
            Role: Test Expert
            Task: Analyze the following content.
            
            Steps:
            - Read carefully
            - Identify key elements
            - Provide analysis
            
            Content: This is test content.
            """
        )
        
        # Create deterministic transition (no randomness)
        self.deterministic_transition = StateTransition(stochasticity=0.0)
        
        # Create stochastic transition (with randomness)
        self.stochastic_transition = StateTransition(stochasticity=0.5)
    
    def test_deterministic_transition(self):
        """Test deterministic state transitions."""
        # Create action
        action = AddRoleAction(parameters={"role_text": "New Expert"})
        
        # Apply transition
        new_state = self.deterministic_transition.apply(self.basic_state, action)
        
        # Check that action was applied correctly
        assert "New Expert" in new_state.text
        assert "Test Expert" not in new_state.text
        assert new_state.components["role"] == "New Expert"
        
        # Test with non-applicable action
        # Create an action that's already been applied
        action = AddRoleAction(parameters={"role_text": "New Expert"})
        
        # Apply transition again
        same_state = self.deterministic_transition.apply(new_state, action)
        
        # Action shouldn't be applied (not applicable)
        assert same_state == new_state
    
    def test_stochastic_transition(self):
        """Test stochastic state transitions."""
        # Create action
        action = ModifyWorkflowAction(parameters={
            "steps": ["New Step 1", "New Step 2"],
            "operation": "replace"
        })
        
        # Apply transition multiple times to test stochasticity
        variations = set()
        for _ in range(10):
            new_state = self.stochastic_transition.apply(self.basic_state, action)
            variations.add(new_state.text)
        
        # With stochasticity enabled, we should get at least some variations
        # Note: This test might occasionally fail due to randomness
        assert len(variations) > 1
    
    def test_transition_history(self):
        """Test that transitions correctly track history."""
        # Create action
        action = AddRoleAction(parameters={"role_text": "New Expert"})
        
        # Apply transition
        new_state = self.deterministic_transition.apply(self.basic_state, action)
        
        # Check that history is updated
        assert len(new_state.history) == len(self.basic_state.history) + 1
        assert "add_role" in new_state.history[-1]
        
        # Check that parent reference is set
        assert new_state.parent == self.basic_state
        
        # Check that action is recorded
        assert new_state.action_applied == str(action)