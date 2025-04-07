"""
Tests for the MCTS Node implementation.
"""
import pytest
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mcts.node import MCTSNode

class TestMCTSNode:
    """Test cases for MCTSNode class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basic_state = PromptState("Task: Analyze the sentiment of the text.")
        self.root_node = MCTSNode(self.basic_state)
        
        # Create some test actions
        self.test_action1 = create_action("add_role", parameters={"role_text": "Test Role"})
        self.test_action2 = create_action("add_constraint", parameters={"constraint_text": "Test Constraint"})
    
    def test_node_initialization(self):
        """Test node initialization."""
        # Test basic properties
        assert self.root_node.state == self.basic_state
        assert self.root_node.parent is None
        assert self.root_node.action_applied is None
        assert self.root_node.node_id is not None
        
        # Test statistics initialization
        assert self.root_node.visit_count == 0
        assert self.root_node.total_reward == 0.0
        assert self.root_node.avg_reward == 0.0
        
        # Test children and available actions
        assert len(self.root_node.children) == 0
        assert len(self.root_node.available_actions) == 0
        assert not self.root_node.fully_expanded
        
        # Test generation
        assert self.root_node.generation == 0
    
    def test_add_child(self):
        """Test adding a child node."""
        # Create a new state
        new_state = PromptState("Role: Test Role\nTask: Analyze the sentiment of the text.")
        
        # Add a child
        child = self.root_node.add_child(self.test_action1, new_state)
        
        # Verify child properties
        assert child.state == new_state
        assert child.parent == self.root_node
        assert child.action_applied == self.test_action1
        assert child.generation == 1
        
        # Verify root node updated
        assert len(self.root_node.children) == 1
        assert str(self.test_action1) in self.root_node.children
        assert self.root_node.children[str(self.test_action1)] == child
        
        # Add another child
        another_state = PromptState("Task: Analyze the sentiment of the text.\nConstraint: Test Constraint")
        another_child = self.root_node.add_child(self.test_action2, another_state)
        
        # Verify multiple children
        assert len(self.root_node.children) == 2
        assert str(self.test_action2) in self.root_node.children
        assert self.root_node.children[str(self.test_action2)] == another_child
    
    def test_update_statistics(self):
        """Test updating node statistics."""
        # Initial values
        assert self.root_node.visit_count == 0
        assert self.root_node.total_reward == 0.0
        assert self.root_node.avg_reward == 0.0
        
        # First update
        self.root_node.update_statistics(0.5)
        assert self.root_node.visit_count == 1
        assert self.root_node.total_reward == 0.5
        assert self.root_node.avg_reward == 0.5
        
        # Second update
        self.root_node.update_statistics(0.7)
        assert self.root_node.visit_count == 2
        assert self.root_node.total_reward == 1.2
        assert self.root_node.avg_reward == 0.6
        
        # Third update
        self.root_node.update_statistics(0.9)
        assert self.root_node.visit_count == 3
        assert self.root_node.total_reward == 2.1
        assert self.root_node.avg_reward == 0.7
    
    def test_set_available_actions(self):
        """Test setting available actions."""
        # Create actions
        actions = [
            create_action("add_role", parameters={"role_text": "Role"}),
            create_action("add_goal", parameters={"goal_text": "Goal"}),
            create_action("add_constraint", parameters={"constraint_text": "Constraint"})
        ]
        
        # Set available actions
        self.root_node.set_available_actions(actions)
        
        # Verify available actions
        assert len(self.root_node.available_actions) == 3
        for action in actions:
            assert str(action) in self.root_node.available_actions
    
    def test_mark_fully_expanded(self):
        """Test marking a node as fully expanded."""
        # Initially not fully expanded
        assert not self.root_node.fully_expanded
        
        # Mark as fully expanded
        self.root_node.mark_fully_expanded()
        
        # Verify
        assert self.root_node.fully_expanded
    
    def test_is_leaf(self):
        """Test leaf node detection."""
        # Root is initially a leaf
        assert self.root_node.is_leaf()
        
        # Add a child
        new_state = PromptState("Role: Test Role\nTask: Analyze the sentiment of the text.")
        self.root_node.add_child(self.test_action1, new_state)
        
        # Root is no longer a leaf
        assert not self.root_node.is_leaf()
    
    def test_is_fully_expanded(self):
        """Test fully expanded detection."""
        # Initially not fully expanded
        assert not self.root_node.is_fully_expanded()
        
        # Set some available actions
        actions = [
            create_action("add_role", parameters={"role_text": "Role"}),
            create_action("add_goal", parameters={"goal_text": "Goal"})
        ]
        self.root_node.set_available_actions(actions)
        
        # Still not fully expanded (has available actions)
        assert not self.root_node.is_fully_expanded()
        
        # Remove all available actions
        self.root_node.available_actions = set()
        
        # Now fully expanded (no available actions)
        assert self.root_node.is_fully_expanded()
        
        # Explicit marking overrides
        self.root_node.available_actions = set(str(action) for action in actions)
        self.root_node.mark_fully_expanded()
        assert self.root_node.is_fully_expanded()
    
    def test_add_evolution_operation(self):
        """Test adding evolution operations."""
        # Initially empty
        assert len(self.root_node.evolution_history) == 0
        
        # Add operations
        self.root_node.add_evolution_operation("mutation")
        self.root_node.add_evolution_operation("crossover with node123")
        
        # Verify
        assert len(self.root_node.evolution_history) == 2
        assert "mutation" in self.root_node.evolution_history
        assert "crossover with node123" in self.root_node.evolution_history
    
    def test_add_error_feedback(self):
        """Test adding error feedback."""
        # Initially empty
        assert len(self.root_node.error_feedback) == 0
        
        # Add feedback
        feedback = {
            "type": "format_error",
            "description": "Missing output format",
            "suggestion": "Add output format specification"
        }
        self.root_node.add_error_feedback(feedback)
        
        # Verify
        assert len(self.root_node.error_feedback) == 1
        assert self.root_node.error_feedback[0] == feedback
    
    def test_get_path_from_root(self):
        """Test getting path from root."""
        # Create a chain of nodes
        state1 = PromptState("Role: Test Role\nTask: Analyze the sentiment of the text.")
        child1 = self.root_node.add_child(self.test_action1, state1)
        
        state2 = PromptState("Role: Test Role\nTask: Analyze the sentiment of the text.\nConstraint: Test Constraint")
        child2 = child1.add_child(self.test_action2, state2)
        
        # Get path from root for different nodes
        root_path = self.root_node.get_path_from_root()
        child1_path = child1.get_path_from_root()
        child2_path = child2.get_path_from_root()
        
        # Verify paths
        assert root_path == [self.root_node]
        assert child1_path == [self.root_node, child1]
        assert child2_path == [self.root_node, child1, child2]
    
    def test_get_action_path_from_root(self):
        """Test getting action path from root."""
        # Create a chain of nodes
        state1 = PromptState("Role: Test Role\nTask: Analyze the sentiment of the text.")
        child1 = self.root_node.add_child(self.test_action1, state1)
        
        state2 = PromptState("Role: Test Role\nTask: Analyze the sentiment of the text.\nConstraint: Test Constraint")
        child2 = child1.add_child(self.test_action2, state2)
        
        # Get action path from root for different nodes
        root_path = self.root_node.get_action_path_from_root()
        child1_path = child1.get_action_path_from_root()
        child2_path = child2.get_action_path_from_root()
        
        # Verify paths
        assert root_path == []
        assert child1_path == [self.test_action1]
        assert child2_path == [self.test_action1, self.test_action2]