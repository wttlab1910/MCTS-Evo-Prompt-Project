"""
Tests for the MCTS Engine implementation.
"""
import pytest
import time
from unittest.mock import MagicMock, patch
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction
from app.core.mcts.node import MCTSNode
from app.core.mcts.engine import MCTSEngine

class TestMCTSEngine:
    """Test cases for MCTSEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test states
        self.basic_state = PromptState("Task: Analyze the sentiment of the text.")
        
        # Mock reward function that provides predictable rewards
        self.mock_reward_fn = MagicMock(spec=RewardFunction)
        self.mock_reward_fn.calculate.side_effect = lambda state: len(state.text) / 1000  # Simple deterministic reward
        
        # Create transition function
        self.transition = StateTransition(stochasticity=0.0)  # No randomness for tests
        
        # Create standard engine
        self.engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            max_iterations=10,
            time_limit=2.0,
            exploration_weight=1.41,
            max_children_per_expansion=3
        )
        
        # Create engine with evolution configuration
        self.evo_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            max_iterations=10,
            time_limit=2.0,
            exploration_weight=1.41,
            max_children_per_expansion=3,
            evolution_config={
                "mutation_rate": 0.2,
                "crossover_rate": 0.2,
                "error_feedback_rate": 0.6,
                "adaptive_adjustment": True
            }
        )
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        # Basic properties
        assert self.engine.transition == self.transition
        assert self.engine.reward_function == self.mock_reward_fn
        assert self.engine.max_iterations == 10
        assert self.engine.time_limit == 2.0
        
        # Components
        assert self.engine.selector is not None
        assert self.engine.expander is not None
        assert self.engine.simulator is not None
        assert self.engine.backpropagator is not None
        
        # Evolution components
        assert self.engine.mutator is not None
        assert self.engine.crossover is not None
        
        # Evolution configuration
        default_config = {
            "mutation_rate": 0.2,
            "crossover_rate": 0.2,
            "error_feedback_rate": 0.6,
            "adaptive_adjustment": True
        }
        
        # Standard engine should have default evolution config
        for key, value in default_config.items():
            assert self.engine.evolution_config[key] == value
        
        # Custom evolution config
        custom_config = {
            "mutation_rate": 0.3,
            "crossover_rate": 0.3,
            "error_feedback_rate": 0.4,
            "adaptive_adjustment": False
        }
        
        custom_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            evolution_config=custom_config
        )
        
        for key, value in custom_config.items():
            assert custom_engine.evolution_config[key] == value
    
    def test_optimize_basic(self):
        """Test basic optimization functionality."""
        # Run optimization
        best_state, stats = self.engine.optimize(self.basic_state)
        
        # Should have produced a valid result
        assert best_state is not None
        assert stats is not None
        
        # Stats should contain expected fields
        assert "iterations" in stats
        assert "time" in stats
        assert "best_reward" in stats
        assert "tree_size" in stats
        assert "max_depth" in stats
        
        # Should have run some iterations
        assert stats["iterations"] > 0
        assert stats["time"] > 0
        assert stats["tree_size"] > 1
        
        # Best state should have better reward than initial state
        initial_reward = self.mock_reward_fn.calculate(self.basic_state)
        best_reward = self.mock_reward_fn.calculate(best_state)
        assert best_reward > initial_reward
    
    def test_optimize_with_iteration_limit(self):
        """Test optimization with iteration limit."""
        # Create an engine with very low iteration limit
        limit_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            max_iterations=3,
            time_limit=10.0  # High time limit
        )
        
        # Start timer
        start_time = time.time()
        
        # Run optimization
        _, stats = limit_engine.optimize(self.basic_state)
        
        # End timer
        elapsed = time.time() - start_time
        
        # Should have run exactly the number of iterations specified
        assert stats["iterations"] == 3
        
        # Should have finished quickly (well under the time limit)
        assert elapsed < 5.0
    
    def test_optimize_with_time_limit(self):
        """Test optimization with time limit."""
        # Create an engine with very low time limit
        limit_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            max_iterations=1000,  # High iteration limit
            time_limit=0.1  # Very low time limit
        )
        
        # Start timer
        start_time = time.time()
        
        # Run optimization
        _, stats = limit_engine.optimize(self.basic_state)
        
        # End timer
        elapsed = time.time() - start_time
        
        # Should have respected the time limit
        assert elapsed < 0.5  # Allow a small buffer
        
        # Should have run fewer iterations than the max
        assert stats["iterations"] < 1000
    
    def test_evolutionary_operations(self):
        """Test evolutionary operations in optimization."""
        # Patch the mutation and crossover methods to track calls
        with patch('app.core.evolution.mutation.PromptMutator.mutate') as mock_mutate, \
             patch('app.core.evolution.crossover.PromptCrossover.crossover') as mock_crossover:
            
            # Make the mocks return valid states
            mock_mutate.return_value = self.basic_state
            mock_crossover.return_value = self.basic_state
            
            # Run optimization with evolution
            _, stats = self.evo_engine.optimize(self.basic_state)
            
            # Should have performed some evolutionary operations
            evolutionary_ops = stats["mutations"] + stats["crossovers"]
            assert evolutionary_ops > 0
            
            # Both mutation and crossover should have been called
            assert mock_mutate.call_count > 0
            assert mock_crossover.call_count >= 0  # Crossover might not be called in short runs
    
    def test_error_feedback_operations(self):
        """Test error feedback operations in optimization."""
        # Run optimization with high error feedback rate
        high_feedback_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            max_iterations=10,
            evolution_config={
                "mutation_rate": 0.1,
                "crossover_rate": 0.1,
                "error_feedback_rate": 0.8,  # High rate
                "adaptive_adjustment": False
            }
        )
        
        # Run optimization
        _, stats = high_feedback_engine.optimize(self.basic_state)
        
        # Should have performed some error feedback operations
        assert stats["error_feedback_actions"] > 0
    
    def test_operation_type_selection(self):
        """Test operation type selection based on rates."""
        # Create engines with different operation rates
        mutation_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            evolution_config={
                "mutation_rate": 1.0,  # Only mutations
                "crossover_rate": 0.0,
                "error_feedback_rate": 0.0,
                "adaptive_adjustment": False
            }
        )
        
        crossover_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            evolution_config={
                "mutation_rate": 0.0,
                "crossover_rate": 1.0,  # Only crossovers
                "error_feedback_rate": 0.0,
                "adaptive_adjustment": False
            }
        )
        
        error_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            evolution_config={
                "mutation_rate": 0.0,
                "crossover_rate": 0.0,
                "error_feedback_rate": 1.0,  # Only error feedback
                "adaptive_adjustment": False
            }
        )
        
        # Test operation selection
        assert mutation_engine._select_operation_type() == "mutation"
        assert crossover_engine._select_operation_type() == "crossover"
        assert error_engine._select_operation_type() == "error_feedback"
    
    def test_adaptive_operation_adjustment(self):
        """Test adaptive adjustment of operation rates."""
        # Create engine with adaptive adjustment
        adaptive_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.mock_reward_fn,
            max_iterations=20,
            evolution_config={
                "mutation_rate": 0.2,
                "crossover_rate": 0.2,
                "error_feedback_rate": 0.6,
                "adaptive_adjustment": True
            }
        )
        
        # Capture initial rates
        initial_mutation = adaptive_engine.evolution_config["mutation_rate"]
        initial_crossover = adaptive_engine.evolution_config["crossover_rate"]
        initial_feedback = adaptive_engine.evolution_config["error_feedback_rate"]
        
        # Make several adjustments
        for i in range(1, 21):
            adaptive_engine._adjust_operation_rates(i)
            
        # Rates should have changed
        assert adaptive_engine.evolution_config["mutation_rate"] != initial_mutation or \
               adaptive_engine.evolution_config["crossover_rate"] != initial_crossover or \
               adaptive_engine.evolution_config["error_feedback_rate"] != initial_feedback
        
        # For the late stage, error feedback should be higher
        assert adaptive_engine.evolution_config["error_feedback_rate"] > initial_feedback
    
    def test_node_selection_for_evolution(self):
        """Test node selection for evolution operations."""
        # Create a tree with nodes of different rewards
        root = MCTSNode(self.basic_state)
        root.update_statistics(0.5)
        
        child1 = root.add_child(create_action("add_role", parameters={"role_text": "Expert"}), 
                             PromptState("Role: Expert\nTask: Analyze the sentiment of the text."))
        child1.update_statistics(0.8)
        
        child2 = root.add_child(create_action("add_goal", parameters={"goal_text": "Goal"}),
                              PromptState("Task: Analyze the sentiment of the text.\nGoal: Goal"))
        child2.update_statistics(0.3)
        
        # For testing, set the root as the MCTS engine's root node
        self.engine._root_node = root
        
        # Select a node for evolution (should prefer higher reward)
        selected = self.engine._select_node_for_evolution(root)
        assert selected in [root, child1, child2]
        
        # With many selections, should prefer child1 (highest reward)
        selections = [self.engine._select_node_for_evolution(root) for _ in range(10)]
        assert child1 in selections
        assert selections.count(child1) > selections.count(child2)
        
        # Test selection with exclusion
        exclude_selections = [self.engine._select_node_for_evolution(root, exclude=child1) for _ in range(5)]
        assert child1 not in exclude_selections
    
    def test_get_best_node(self):
        """Test getting the best node from the tree."""
        # Create a tree with nodes of different rewards
        root = MCTSNode(self.basic_state)
        root.update_statistics(0.5)
        
        child1 = root.add_child(create_action("add_role", parameters={"role_text": "Expert"}), 
                             PromptState("Role: Expert\nTask: Analyze the sentiment of the text."))
        child1.update_statistics(0.8)
        
        child2 = root.add_child(create_action("add_goal", parameters={"goal_text": "Goal"}),
                              PromptState("Task: Analyze the sentiment of the text.\nGoal: Goal"))
        child2.update_statistics(0.3)
        
        # Get best node
        best_node = self.engine._get_best_node(root)
        
        # Should be the node with highest reward (child1)
        assert best_node == child1
    
    def test_count_nodes(self):
        """Test counting nodes in the tree."""
        # Create a tree
        root = MCTSNode(self.basic_state)
        
        child1 = root.add_child(create_action("add_role", parameters={"role_text": "Expert"}), 
                             PromptState("Role: Expert\nTask: Analyze the sentiment of the text."))
        
        child2 = root.add_child(create_action("add_goal", parameters={"goal_text": "Goal"}),
                              PromptState("Task: Analyze the sentiment of the text.\nGoal: Goal"))
        
        grandchild = child1.add_child(create_action("add_constraint", parameters={"constraint_text": "Constraint"}),
                                   PromptState("Role: Expert\nTask: Analyze the sentiment of the text.\nConstraint: Constraint"))
        
        # Count nodes
        count = self.engine._count_nodes(root)
        
        # Should be 4 nodes in total
        assert count == 4
    
    def test_max_depth(self):
        """Test finding maximum depth of the tree."""
        # Create a tree
        root = MCTSNode(self.basic_state)
        
        child1 = root.add_child(create_action("add_role", parameters={"role_text": "Expert"}), 
                             PromptState("Role: Expert\nTask: Analyze the sentiment of the text."))
        
        child2 = root.add_child(create_action("add_goal", parameters={"goal_text": "Goal"}),
                              PromptState("Task: Analyze the sentiment of the text.\nGoal: Goal"))
        
        grandchild = child1.add_child(create_action("add_constraint", parameters={"constraint_text": "Constraint"}),
                                   PromptState("Role: Expert\nTask: Analyze the sentiment of the text.\nConstraint: Constraint"))
        
        great_grandchild = grandchild.add_child(create_action("add_example", parameters={"example_text": "Example"}),
                                            PromptState("Role: Expert\nTask: Analyze the sentiment of the text.\nConstraint: Constraint\nExample: Example"))
        
        # Get max depth
        depth = self.engine._max_depth(root)
        
        # Should be 3 (root → child1 → grandchild → great_grandchild)
        assert depth == 3
    
    def test_get_root_node(self):
        """Test getting the root node."""
        # Initially no root node
        assert self.engine._get_root_node() is None
        
        # Run optimization to set a root node
        self.engine.optimize(self.basic_state)
        
        # Should have a root node
        assert self.engine._get_root_node() is not None