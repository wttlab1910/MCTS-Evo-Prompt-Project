"""
Tests for the MCTS Backpropagation implementation.
"""
import pytest
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mcts.node import MCTSNode
from app.core.mcts.backprop import Backpropagator

class TestMCTSBackprop:
    """Test cases for Backpropagator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a basic MCTS tree for testing
        self.root_state = PromptState("Task: Analyze the sentiment of the text.")
        self.root = MCTSNode(self.root_state)
        
        # Create level 1 children
        self.action1 = create_action("add_role", parameters={"role_text": "Expert"})
        self.state1 = PromptState("Role: Expert\nTask: Analyze the sentiment of the text.")
        self.child1 = self.root.add_child(self.action1, self.state1)
        
        self.action2 = create_action("modify_workflow", parameters={"steps": ["Step 1", "Step 2"]})
        self.state2 = PromptState("Task: Analyze the sentiment of the text.\nSteps:\n- Step 1\n- Step 2")
        self.child2 = self.root.add_child(self.action2, self.state2)
        
        # Create level 2 children (grandchildren)
        self.action3 = create_action("add_constraint", parameters={"constraint_text": "Be concise"})
        self.state3 = PromptState("Role: Expert\nTask: Analyze the sentiment of the text.\nConstraint: Be concise")
        self.grandchild1 = self.child1.add_child(self.action3, self.state3)
        
        # Create backpropagators with different discount factors
        self.no_discount_backprop = Backpropagator(discount_factor=1.0)
        self.discount_backprop = Backpropagator(discount_factor=0.9)
    
    def test_backpropagator_initialization(self):
        """Test backpropagator initialization."""
        # Default values
        default_backprop = Backpropagator()
        assert default_backprop.discount_factor == 1.0
        
        # Custom values
        custom_backprop = Backpropagator(discount_factor=0.5)
        assert custom_backprop.discount_factor == 0.5
    
    def test_backpropagate_leaf_to_root(self):
        """Test backpropagation from leaf to root."""
        # Initial visit counts and rewards
        assert self.root.visit_count == 0
        assert self.child1.visit_count == 0
        assert self.grandchild1.visit_count == 0
        
        # Backpropagate from grandchild to root
        reward = 0.8
        self.no_discount_backprop.backpropagate(self.grandchild1, reward)
        
        # All nodes in the path should have been updated
        assert self.grandchild1.visit_count == 1
        assert self.grandchild1.total_reward == reward
        assert self.grandchild1.avg_reward == reward
        
        assert self.child1.visit_count == 1
        assert self.child1.total_reward == reward
        assert self.child1.avg_reward == reward
        
        assert self.root.visit_count == 1
        assert self.root.total_reward == reward
        assert self.root.avg_reward == reward
        
        # Nodes not in the path should remain unchanged
        assert self.child2.visit_count == 0
        assert self.child2.total_reward == 0.0
        assert self.child2.avg_reward == 0.0
    
    def test_backpropagate_with_discount(self):
        """Test backpropagation with discount factor."""
        # Backpropagate with discount
        reward = 1.0
        self.discount_backprop.backpropagate(self.grandchild1, reward)
        
        # Verify discounted rewards
        assert self.grandchild1.visit_count == 1
        assert self.grandchild1.total_reward == reward
        assert self.grandchild1.avg_reward == reward
        
        # Child1 gets discounted reward (reward * 0.9)
        assert self.child1.visit_count == 1
        assert pytest.approx(self.child1.total_reward) == reward * 0.9
        assert pytest.approx(self.child1.avg_reward) == reward * 0.9
        
        # Root gets double-discounted reward (reward * 0.9 * 0.9)
        assert self.root.visit_count == 1
        assert pytest.approx(self.root.total_reward) == reward * 0.9 * 0.9
        assert pytest.approx(self.root.avg_reward) == reward * 0.9 * 0.9
    
    def test_backpropagate_multiple_times(self):
        """Test backpropagation multiple times."""
        # Backpropagate multiple times with different rewards
        rewards = [0.6, 0.7, 0.9]
        for reward in rewards:
            self.no_discount_backprop.backpropagate(self.grandchild1, reward)
        
        # Verify correct updates
        assert self.grandchild1.visit_count == len(rewards)
        assert self.grandchild1.total_reward == sum(rewards)
        assert self.grandchild1.avg_reward == sum(rewards) / len(rewards)
        
        assert self.child1.visit_count == len(rewards)
        assert self.child1.total_reward == sum(rewards)
        assert self.child1.avg_reward == sum(rewards) / len(rewards)
        
        assert self.root.visit_count == len(rewards)
        assert self.root.total_reward == sum(rewards)
        assert self.root.avg_reward == sum(rewards) / len(rewards)
    
    def test_backpropagate_with_path(self):
        """Test backpropagation with explicit path."""
        # Create path from leaf to root
        path = [self.grandchild1, self.child1, self.root]
        
        # Backpropagate with path
        reward = 0.8
        self.no_discount_backprop.backpropagate_with_path(path, reward)
        
        # All nodes in the path should have been updated
        assert self.grandchild1.visit_count == 1
        assert self.grandchild1.total_reward == reward
        assert self.grandchild1.avg_reward == reward
        
        assert self.child1.visit_count == 1
        assert self.child1.total_reward == reward
        assert self.child1.avg_reward == reward
        
        assert self.root.visit_count == 1
        assert self.root.total_reward == reward
        assert self.root.avg_reward == reward
    
    def test_backpropagate_with_path_and_discount(self):
        """Test backpropagation with path and discount factor."""
        # Create path from leaf to root
        path = [self.grandchild1, self.child1, self.root]
        
        # Backpropagate with path and discount
        reward = 1.0
        self.discount_backprop.backpropagate_with_path(path, reward)
        
        # Verify discounted rewards
        # First node (grandchild) gets full reward
        assert self.grandchild1.visit_count == 1
        assert self.grandchild1.total_reward == reward
        assert self.grandchild1.avg_reward == reward
        
        # Second node (child) gets discounted reward
        assert self.child1.visit_count == 1
        assert pytest.approx(self.child1.total_reward) == reward * 0.9
        assert pytest.approx(self.child1.avg_reward) == reward * 0.9
        
        # Third node (root) gets double-discounted reward
        assert self.root.visit_count == 1
        assert pytest.approx(self.root.total_reward) == reward * 0.9 * 0.9
        assert pytest.approx(self.root.avg_reward) == reward * 0.9 * 0.9
    
    def test_backpropagate_empty_path(self):
        """Test backpropagation with empty path."""
        # Empty path should not raise errors but do nothing
        self.no_discount_backprop.backpropagate_with_path([], 0.5)
        
        # No nodes should have been updated
        assert self.root.visit_count == 0
        assert self.child1.visit_count == 0
        assert self.grandchild1.visit_count == 0
    
    def test_set_discount_factor(self):
        """Test setting discount factor."""
        # Initial value
        assert self.no_discount_backprop.discount_factor == 1.0
        
        # Set new value
        self.no_discount_backprop.set_discount_factor(0.7)
        
        # Verify
        assert self.no_discount_backprop.discount_factor == 0.7
        
        # Test with the new discount factor
        reward = 1.0
        self.no_discount_backprop.backpropagate(self.grandchild1, reward)
        
        # Verify discounted rewards
        assert pytest.approx(self.child1.total_reward) == reward * 0.7
        assert pytest.approx(self.root.total_reward) == reward * 0.7 * 0.7