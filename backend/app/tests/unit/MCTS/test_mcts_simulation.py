"""
Tests for the MCTS Simulation strategy.
"""
import pytest
from app.core.mdp.state import PromptState
from app.core.mdp.reward import RewardFunction
from app.core.mcts.node import MCTSNode
from app.core.mcts.simulation import PromptSimulator

class TestMCTSSimulation:
    """Test cases for PromptSimulator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test states
        self.empty_state = PromptState("")
        
        self.basic_state = PromptState(
            """
            Task: Analyze the sentiment of the text.
            """
        )
        
        self.structured_state = PromptState(
            """
            Role: Sentiment Analysis Expert
            Task: Analyze the sentiment of the provided text.
            
            Steps:
            - Read the text carefully
            - Identify sentiment-bearing words and phrases
            - Determine overall sentiment
            
            Output Format: Provide sentiment as positive, negative, or neutral.
            """
        )
        
        # Create test nodes
        self.empty_node = MCTSNode(self.empty_state)
        self.basic_node = MCTSNode(self.basic_state)
        self.structured_node = MCTSNode(self.structured_state)
        
        # Create a basic reward function
        self.reward_fn = RewardFunction()
        
        # Create simulators
        self.basic_simulator = PromptSimulator(reward_function=self.reward_fn)
        
        # Create simulator with custom evaluator
        self.custom_evaluator = lambda state: 0.5 if not state.text.strip() else 0.8
        self.custom_simulator = PromptSimulator(
            reward_function=self.reward_fn,
            custom_evaluator=self.custom_evaluator
        )
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        # Basic simulator
        assert self.basic_simulator.reward_function == self.reward_fn
        assert self.basic_simulator.depth_limit == 0  # Default
        assert self.basic_simulator.custom_evaluator is None
        
        # Custom simulator
        assert self.custom_simulator.reward_function == self.reward_fn
        assert self.custom_simulator.custom_evaluator == self.custom_evaluator
        
        # With depth limit
        depth_simulator = PromptSimulator(reward_function=self.reward_fn, depth_limit=3)
        assert depth_simulator.depth_limit == 3
    
    def test_simulate_basic(self):
        """Test basic simulation."""
        # Simulate empty node
        empty_reward = self.basic_simulator.simulate(self.empty_node)
        
        # Simulate basic node
        basic_reward = self.basic_simulator.simulate(self.basic_node)
        
        # Simulate structured node
        structured_reward = self.basic_simulator.simulate(self.structured_node)
        
        # Rewards should be in reasonable range
        assert 0 <= empty_reward <= 1
        assert 0 <= basic_reward <= 1
        assert 0 <= structured_reward <= 1
        
        # More structured prompts should have higher rewards
        assert empty_reward < basic_reward
        assert basic_reward < structured_reward
    
    def test_simulate_with_custom_evaluator(self):
        """Test simulation with custom evaluator."""
        # Simulate with custom evaluator
        empty_reward = self.custom_simulator.simulate(self.empty_node)
        structured_reward = self.custom_simulator.simulate(self.structured_node)
        
        # Custom evaluator returns 0.5 for empty and 0.8 for non-empty
        assert empty_reward == 0.5
        assert structured_reward == 0.8
    
    def test_evaluate_state(self):
        """Test direct state evaluation."""
        # Evaluate states
        empty_reward = self.basic_simulator.evaluate_state(self.empty_state)
        basic_reward = self.basic_simulator.evaluate_state(self.basic_state)
        structured_reward = self.basic_simulator.evaluate_state(self.structured_state)
        
        # Rewards should be in reasonable range
        assert 0 <= empty_reward <= 1
        assert 0 <= basic_reward <= 1
        assert 0 <= structured_reward <= 1
        
        # More structured prompts should have higher rewards
        assert empty_reward < basic_reward
        assert basic_reward < structured_reward
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        # Evaluate batch of states
        states = [self.empty_state, self.basic_state, self.structured_state]
        rewards = self.basic_simulator.evaluate_batch(states)
        
        # Should return rewards for all states
        assert len(rewards) == len(states)
        
        # Rewards should be in reasonable range
        assert all(0 <= reward <= 1 for reward in rewards)
        
        # More structured prompts should have higher rewards
        assert rewards[0] < rewards[1] < rewards[2]
    
    def test_custom_evaluator_with_batch(self):
        """Test batch evaluation with custom evaluator."""
        # Evaluate batch with custom evaluator
        states = [self.empty_state, self.basic_state, self.structured_state]
        rewards = self.custom_simulator.evaluate_batch(states)
        
        # Should return rewards for all states
        assert len(rewards) == len(states)
        
        # Custom evaluator returns 0.5 for empty and 0.8 for non-empty
        assert rewards[0] == 0.5
        assert rewards[1] == 0.8
        assert rewards[2] == 0.8
    
    def test_simulation_equivalent_to_evaluate(self):
        """Test that simulation and direct evaluation are equivalent."""
        # For depth_limit=0, simulation and evaluation should give same results
        simulate_reward = self.basic_simulator.simulate(self.basic_node)
        evaluate_reward = self.basic_simulator.evaluate_state(self.basic_state)
        
        # Results should be identical
        assert simulate_reward == evaluate_reward