"""
Tests for the MCTS Expansion strategy.
"""
import pytest
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mdp.transition import StateTransition
from app.core.mcts.node import MCTSNode
from app.core.mcts.expansion import ActionExpander

class TestMCTSExpansion:
    """Test cases for ActionExpander class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basic_state = PromptState("Task: Analyze the sentiment of the text.")
        self.root_node = MCTSNode(self.basic_state)
        
        # Create transition function
        self.transition = StateTransition(stochasticity=0.0)  # No randomness for tests
        
        # Create standard expander
        self.expander = ActionExpander(transition=self.transition, max_children_per_expansion=3)
        
        # Create limited expander
        self.limited_expander = ActionExpander(transition=self.transition, max_children_per_expansion=1)
        
        # Create actions
        self.actions = [
            create_action("add_role", parameters={"role_text": "Expert"}),
            create_action("add_goal", parameters={"goal_text": "Determine sentiment"}),
            create_action("add_constraint", parameters={"constraint_text": "Be concise"}),
            create_action("add_explanation", parameters={"explanation_text": "Explanation", "target": "task"})
        ]
    
    def test_expander_initialization(self):
        """Test expander initialization."""
        # Default values
        default_expander = ActionExpander(transition=self.transition)
        assert default_expander.transition == self.transition
        assert default_expander.max_children_per_expansion == 5  # Default value
        assert default_expander.action_filter is None
        
        # Custom values
        custom_expander = ActionExpander(
            transition=self.transition,
            max_children_per_expansion=2,
            action_filter=lambda state, action: "role" in str(action)
        )
        assert custom_expander.transition == self.transition
        assert custom_expander.max_children_per_expansion == 2
        assert custom_expander.action_filter is not None
    
    def test_expand_with_available_actions(self):
        """Test expansion with available actions."""
        # Expand with actions
        children = self.expander.expand(self.root_node, self.actions)
        
        # Should have created children
        assert len(children) == 3  # Limited by max_children_per_expansion
        assert len(self.root_node.children) == 3
        
        # Verify each child has correct properties
        for child in children:
            assert child.parent == self.root_node
            assert child.state != self.root_node.state
            assert str(child.action_applied) in self.root_node.children
            assert self.root_node.children[str(child.action_applied)] == child
    
    def test_expand_with_limited_expansion(self):
        """Test expansion with limited child count."""
        # Expand with limited expander
        children = self.limited_expander.expand(self.root_node, self.actions)
        
        # Should have created only one child
        assert len(children) == 1
        assert len(self.root_node.children) == 1
    
    def test_expand_fully_expanded_node(self):
        """Test expanding a fully expanded node."""
        # Mark node as fully expanded
        self.root_node.mark_fully_expanded()
        
        # Try to expand
        children = self.expander.expand(self.root_node, self.actions)
        
        # Should not create any children
        assert len(children) == 0
        assert len(self.root_node.children) == 0
    
    def test_expand_with_action_filter(self):
        """Test expansion with action filter."""
        # Create custom expander with filter
        filter_expander = ActionExpander(
            transition=self.transition,
            max_children_per_expansion=3,
            action_filter=lambda state, action: "role" in str(action)
        )
        
        # Expand with filter
        children = filter_expander.expand(self.root_node, self.actions)
        
        # Should only create children for actions that pass the filter
        assert len(children) == 1  # Only the "add_role" action
        assert len(self.root_node.children) == 1
        assert str(self.actions[0]) in self.root_node.children
    
    def test_expand_with_empty_actions(self):
        """Test expansion with empty action list."""
        # Expand with empty actions
        children = self.expander.expand(self.root_node, [])
        
        # Should not create any children
        assert len(children) == 0
        assert len(self.root_node.children) == 0
    
    def test_expand_with_action(self):
        """Test expanding with a specific action."""
        # Expand with one action
        action = self.actions[0]
        child = self.expander.expand_with_action(self.root_node, action)
        
        # Should have created a child
        assert child is not None
        assert child.parent == self.root_node
        assert child.action_applied == action
        assert child.state != self.root_node.state
        assert str(action) in self.root_node.children
        assert self.root_node.children[str(action)] == child
    
    def test_expand_with_already_applied_action(self):
        """Test expanding with an already applied action."""
        # Apply action once
        action = self.actions[0]
        first_child = self.expander.expand_with_action(self.root_node, action)
        
        # Try to apply same action again
        second_child = self.expander.expand_with_action(self.root_node, action)
        
        # Should return the existing child, not create a new one
        assert second_child == first_child
        assert len(self.root_node.children) == 1
    
    def test_expand_with_inapplicable_action(self):
        """Test expanding with an inapplicable action."""
        # Create a node with a role already set
        state_with_role = PromptState("Role: Expert\nTask: Analyze the sentiment of the text.")
        node_with_role = MCTSNode(state_with_role)
        
        # Try to apply action that adds a role (should be inapplicable)
        action = create_action("add_role", parameters={"role_text": "Expert"})
        child = self.expander.expand_with_action(node_with_role, action)
        
        # Should not create a child
        assert child is None
        assert len(node_with_role.children) == 0
    
    def test_available_actions_update(self):
        """Test available actions update after expansion."""
        # Set available actions
        self.root_node.set_available_actions(self.actions)
        initial_action_count = len(self.root_node.available_actions)
        
        # Expand with some actions
        children = self.expander.expand(self.root_node, self.actions[:2])
        
        # Available actions should be reduced
        assert len(self.root_node.available_actions) == initial_action_count - len(children)
        
        # Expand until no actions left
        remaining_actions = list(self.root_node.available_actions)
        remaining_children = self.expander.expand(self.root_node, [create_action(*action.split(".", 1)) for action in remaining_actions])
        
        # Should have no available actions left
        assert len(self.root_node.available_actions) == 0
        
        # Should be marked as fully expanded
        assert self.root_node.fully_expanded
    
    def test_generate_actions_for_node(self):
        """Test generating actions for a node."""
        # Generate actions for basic node
        actions = self.expander.generate_actions_for_node(self.root_node)
        
        # Should generate some actions
        assert len(actions) > 0
        
        # Create a more complete state
        complete_state = PromptState("""
        Role: Expert
        Task: Analyze the sentiment of the text.
        Steps:
        - Read the text
        - Identify sentiment words
        - Determine overall sentiment
        Output Format: Positive, Negative, or Neutral
        """)
        complete_node = MCTSNode(complete_state)
        
        # Generate actions for complete node
        complete_actions = self.expander.generate_actions_for_node(complete_node)
        
        # Should generate fewer actions (as many components already exist)
        assert len(complete_actions) < len(actions)
    
    def test_action_applicability(self):
        """Test action applicability checking."""
        # Check applicability of add_role to a node without role
        action = create_action("add_role", parameters={"role_text": "Expert"})
        assert action.is_applicable(self.basic_state)
        
        # Apply the action
        new_state = self.transition.apply(self.basic_state, action)
        
        # Action should no longer be applicable
        assert not action.is_applicable(new_state)
        
        # But a different role could be applicable with replace=False
        append_action = create_action("add_role", parameters={"role_text": "Another Expert", "replace": False})
        assert append_action.is_applicable(new_state)