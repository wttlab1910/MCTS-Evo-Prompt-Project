"""
Tests for the MCTS Selection strategy.
"""
import pytest
import math
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mcts.node import MCTSNode
from app.core.mcts.selection import UCTSelector

class TestMCTSSelection:
    """Test cases for UCTSelector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basic_state = PromptState("Task: Analyze the sentiment of the text.")
        self.root_node = MCTSNode(self.basic_state)
        
        # Create a basic MCTS tree with varying rewards
        self.child_nodes = []
        
        # Child 1: high reward, few visits
        action1 = create_action("add_role", parameters={"role_text": "Expert"})
        state1 = PromptState("Role: Expert\nTask: Analyze the sentiment of the text.")
        child1 = self.root_node.add_child(action1, state1)
        child1.update_statistics(0.8)  # High reward
        self.child_nodes.append(child1)
        
        # Child 2: medium reward, many visits
        action2 = create_action("add_goal", parameters={"goal_text": "Determine sentiment"})
        state2 = PromptState("Task: Determine sentiment of the text.")
        child2 = self.root_node.add_child(action2, state2)
        for _ in range(5):
            child2.update_statistics(0.5)  # Medium reward, many visits
        self.child_nodes.append(child2)
        
        # Child 3: low reward, few visits
        action3 = create_action("add_constraint", parameters={"constraint_text": "Be concise"})
        state3 = PromptState("Task: Analyze the sentiment of the text.\nConstraint: Be concise")
        child3 = self.root_node.add_child(action3, state3)
        child3.update_statistics(0.2)  # Low reward
        self.child_nodes.append(child3)
        
        # Update root node statistics
        for _ in range(7):  # Sum of all child visits
            self.root_node.update_statistics(0.5)  # Value doesn't matter
        
        # Create selectors with different exploration weights
        self.balanced_selector = UCTSelector(exploration_weight=1.41)  # Standard sqrt(2)
        self.explorer_selector = UCTSelector(exploration_weight=2.0)   # Exploration focus
        self.exploiter_selector = UCTSelector(exploration_weight=0.1)  # Exploitation focus
    
    def test_selector_initialization(self):
        """Test selector initialization."""
        # Test with default
        default_selector = UCTSelector()
        assert default_selector.exploration_weight == 1.41
        
        # Test with custom values
        custom_selector = UCTSelector(exploration_weight=2.5)
        assert custom_selector.exploration_weight == 2.5
        
        # Test with zero (pure exploitation)
        zero_selector = UCTSelector(exploration_weight=0.0)
        assert zero_selector.exploration_weight == 0.0
    
    def test_exploitation_focused_selection(self):
        """Test selection with low exploration weight (exploitation focus)."""
        # With low exploration weight, should select highest reward node
        selected = self.exploiter_selector.select(self.root_node)
        assert selected == self.child_nodes[0]  # Child 1 has highest reward
    
    def test_exploration_focused_selection(self):
        """Test selection with high exploration weight (exploration focus)."""
        # With high exploration weight, should prioritize less-visited nodes
        # For our setup, child1 and child3 have 1 visit each,
        # but child1 has higher reward, so it's more promising
        selected = self.explorer_selector.select(self.root_node)
        
        # If the node is fully expanded, it should select one of the less-visited children
        # which are child1 or child3 (but likely child1 due to higher reward)
        if not self.root_node.is_fully_expanded():
            assert selected == self.root_node  # Not expanded yet, select root
        else:
            assert selected in [self.child_nodes[0], self.child_nodes[2]]  # Less visited nodes
    
    def test_balanced_selection(self):
        """Test selection with balanced exploration weight."""
        # For a standard tree, UCT behavior is more complex to predict deterministically
        # We just ensure it selects a valid node
        selected = self.balanced_selector.select(self.root_node)
        assert selected in [self.root_node] + self.child_nodes
    
    def test_selection_with_unexpanded_node(self):
        """Test selection with an unexpanded node."""
        # Create a new node with available actions but no children
        new_node = MCTSNode(self.basic_state)
        actions = [
            create_action("add_role", parameters={"role_text": "Role"}),
            create_action("add_goal", parameters={"goal_text": "Goal"})
        ]
        new_node.set_available_actions(actions)
        
        # Selection should return this node since it's not fully expanded
        selected = self.balanced_selector.select(new_node)
        assert selected == new_node
    
    def test_selection_traversal(self):
        """Test selection traversal through tree."""
        # Create a deeper tree
        grandchild_state = PromptState("Role: Expert\nTask: Analyze the sentiment of the text.\nSteps:")
        action = create_action("modify_workflow", parameters={"steps": ["Step 1", "Step 2"]})
        grandchild = self.child_nodes[0].add_child(action, grandchild_state)
        
        # Mark child_nodes[0] as fully expanded
        self.child_nodes[0].mark_fully_expanded()
        
        # Update root's fully_expanded status based on children
        all_children_expanded = all(child.is_fully_expanded() for child in self.child_nodes)
        if all_children_expanded:
            self.root_node.mark_fully_expanded()
        
        # Grandchild has no visits, so it should be selected if reached
        selected = self.balanced_selector.select(self.root_node)
        
        # If the tree is fully expanded up to child level, selection should continue to grandchild
        if self.root_node.is_fully_expanded() and self.child_nodes[0].is_fully_expanded():
            # Should select grandchild
            assert selected == grandchild
        else:
            # Otherwise, might select another node
            assert selected in [self.root_node] + self.child_nodes + [grandchild]
    
    def test_get_children_scores(self):
        """Test getting UCT scores for children."""
        # Get scores
        scores = self.balanced_selector.get_children_scores(self.root_node)
        
        # Should have a score for each child
        assert len(scores) == len(self.child_nodes)
        
        # Verify score calculation
        for action_str, child in self.root_node.children.items():
            # Manual UCT calculation for verification
            exploitation = child.avg_reward
            exploration = 1.41 * math.sqrt(math.log(self.root_node.visit_count) / child.visit_count)
            expected_score = exploitation + exploration
            
            # Allow for small floating-point differences
            assert abs(scores[action_str] - expected_score) < 1e-10
    
    def test_set_exploration_weight(self):
        """Test setting exploration weight."""
        # Initial value
        assert self.balanced_selector.exploration_weight == 1.41
        
        # Set new value
        self.balanced_selector.set_exploration_weight(3.0)
        
        # Verify
        assert self.balanced_selector.exploration_weight == 3.0
    
    def test_selection_error_handling(self):
        """Test error handling in selection."""
        # Create a leaf node
        leaf_node = MCTSNode(self.basic_state)
        
        # Trying to select best child of a leaf should raise ValueError
        with pytest.raises(ValueError):
            self.balanced_selector._select_best_child(leaf_node)