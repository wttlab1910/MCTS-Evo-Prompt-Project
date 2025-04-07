"""
Comprehensive tests for Phase 3: MCTS Strategic Planning with Evolutionary Algorithms.
"""
import pytest
import random
import time
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction, PerformanceEvaluator
from app.core.mcts.node import MCTSNode
from app.core.mcts.selection import UCTSelector
from app.core.mcts.expansion import ActionExpander
from app.core.mcts.simulation import PromptSimulator
from app.core.mcts.backprop import Backpropagator
from app.core.mcts.engine import MCTSEngine
from app.core.evolution.mutation import PromptMutator
from app.core.evolution.crossover import PromptCrossover
from app.core.evolution.selection import EvolutionSelector
from app.knowledge.error.error_collector import ErrorCollector
from app.knowledge.error.error_analyzer import ErrorAnalyzer
from app.knowledge.error.feedback_generator import FeedbackGenerator

class TestPhase3:
    """
    Tests for Phase 3 components.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
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
        
        # Create transition handler
        self.transition = StateTransition(stochasticity=0.1)
        
        # Create reward function
        self.reward_fn = RewardFunction(
            task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
        )
        
        # Create MCTS components
        self.selector = UCTSelector(exploration_weight=1.41)
        self.expander = ActionExpander(transition=self.transition, max_children_per_expansion=3)
        self.simulator = PromptSimulator(reward_function=self.reward_fn)
        self.backpropagator = Backpropagator()
        
        # Create evolutionary components
        self.mutator = PromptMutator()
        self.crossover = PromptCrossover()
        self.evo_selector = EvolutionSelector()
        
        # Create error feedback components
        self.error_collector = ErrorCollector()
        self.error_analyzer = ErrorAnalyzer()
        self.feedback_generator = FeedbackGenerator()
        
        # Create MCTS engine
        self.mcts_engine = MCTSEngine(
            transition=self.transition,
            reward_function=self.reward_fn,
            max_iterations=20,
            time_limit=5.0,
            exploration_weight=1.41,
            max_children_per_expansion=3
        )
    
    def test_mcts_node_functionality(self):
        """
        Test MCTS node functionality.
        """
        # Create a root node
        root = MCTSNode(state=self.basic_state)
        
        # Add a child node
        action = create_action("add_role", parameters={"role_text": "Test Role"})
        child_state = self.transition.apply(self.basic_state, action)
        child = root.add_child(action, child_state)
        
        # Test node relationships
        assert child.parent == root
        assert str(action) in root.children
        assert root.children[str(action)] == child
        
        # Test node statistics update
        root.update_statistics(0.5)
        root.update_statistics(0.7)
        
        assert root.visit_count == 2
        assert root.total_reward == 1.2
        assert root.avg_reward == 0.6
        
        # Test available actions
        actions = [
            create_action("add_role", parameters={"role_text": "Role 1"}),
            create_action("add_goal", parameters={"goal_text": "Goal 1"}),
            create_action("add_constraint", parameters={"constraint_text": "Constraint 1"})
        ]
        
        root.set_available_actions(actions)
        assert len(root.available_actions) == 3
        
        # Test fully expanded status
        assert not root.is_fully_expanded()
        root.mark_fully_expanded()
        assert root.is_fully_expanded()
        
        # Test path tracking
        path = child.get_path_from_root()
        assert len(path) == 2
        assert path[0] == root
        assert path[1] == child
        
        action_path = child.get_action_path_from_root()
        assert len(action_path) == 1
        assert action_path[0] == action
    
    def test_uct_selection(self):
        """
        Test UCT selection functionality.
        """
        # Create a root node
        root = MCTSNode(state=self.basic_state)
        
        # Add children with different statistics
        actions = [
            create_action("add_role", parameters={"role_text": "Role 1"}),
            create_action("add_goal", parameters={"goal_text": "Goal 1"}),
            create_action("add_constraint", parameters={"constraint_text": "Constraint 1"})
        ]
        
        children = []
        for action in actions:
            child_state = self.transition.apply(self.basic_state, action)
            child = root.add_child(action, child_state)
            children.append(child)
        
        # Update statistics for root and children
        root.update_statistics(0.0)  # Needs at least one visit
        
        children[0].update_statistics(0.7)  # High reward, one visit
        
        children[1].update_statistics(0.6)  # Medium reward, many visits
        children[1].update_statistics(0.6)
        children[1].update_statistics(0.6)
        
        children[2].update_statistics(0.1)  # Low reward, one visit
        
        # With high exploration weight, should select under-explored node
        explorer = UCTSelector(exploration_weight=2.0)
        selected = explorer.select(root)
        
        # Either child 0 or child 2 should be selected (they have fewer visits)
        assert selected in [children[0], children[2]]
        
        # With zero exploration weight, should select highest average reward
        exploiter = UCTSelector(exploration_weight=0.0)
        selected = exploiter.select(root)
        assert selected == children[0]  # Highest average reward
    
    def test_action_expansion(self):
        """
        Test action expansion functionality.
        """
        # Create a root node
        root = MCTSNode(state=self.basic_state)
        
        # Generate some actions
        actions = [
            create_action("add_role", parameters={"role_text": "Role 1"}),
            create_action("add_goal", parameters={"goal_text": "Goal 1"}),
            create_action("add_constraint", parameters={"constraint_text": "Constraint 1"})
        ]
        
        # Expand with some actions
        children = self.expander.expand(root, actions)
        
        # Should create children for all actions (max_children_per_expansion is 3)
        assert len(children) == 3
        assert len(root.children) == 3
        
        # Children should have correct states
        for child in children:
            assert child.parent == root
            assert child.state != root.state
        
        # Create a new node for testing expansion limits
        limited_node = MCTSNode(state=self.basic_state)
        
        # Create a limited expander
        limited_expander = ActionExpander(
            transition=self.transition, 
            max_children_per_expansion=2
        )
        
        # Expand with more actions than the limit
        more_actions = actions + [
            create_action("add_example", parameters={"example_text": "Example 1"})
        ]
        
        limited_children = limited_expander.expand(limited_node, more_actions)
        
        # Should only create 2 children (due to max_children_per_expansion)
        assert len(limited_children) == 2
        assert len(limited_node.children) == 2
    
    def test_simulation_and_backpropagation(self):
        """
        Test simulation and backpropagation functionality.
        """
        # Create a tree with root and one child
        root = MCTSNode(state=self.basic_state)
        
        action = create_action("add_role", parameters={"role_text": "Sentiment Analysis Expert"})
        child_state = self.transition.apply(self.basic_state, action)
        child = root.add_child(action, child_state)
        
        # Simulate the child node
        reward = self.simulator.simulate(child)
        
        # Reward should be higher than zero
        assert reward > 0
        
        # Backpropagate the reward
        self.backpropagator.backpropagate(child, reward)
        
        # Both nodes should have updated statistics
        assert child.visit_count == 1
        assert child.total_reward == reward
        assert child.avg_reward == reward
        
        assert root.visit_count == 1
        assert root.total_reward == reward
        assert root.avg_reward == reward
        
        # Test batch evaluation
        states = [self.basic_state, self.structured_state]
        rewards = self.simulator.evaluate_batch(states)
        
        assert len(rewards) == 2
        assert rewards[1] > rewards[0]  # Structured state should have higher reward
    
    def test_mcts_engine(self):
        """
        Test the MCTS engine with a simple optimization task.
        """
        # Run MCTS optimization
        best_state, stats = self.mcts_engine.optimize(self.basic_state)
        
        # Best state should have better reward than initial state
        initial_reward = self.reward_fn.calculate(self.basic_state)
        best_reward = self.reward_fn.calculate(best_state)
        
        assert best_reward > initial_reward
        
        # Check statistics
        assert stats["iterations"] > 0
        assert stats["time"] > 0
        assert stats["best_reward"] > 0
        assert stats["tree_size"] > 1
        assert stats["max_depth"] > 0
        
        # Best state should have some improvements
        # For example, it might have a role, steps, or output format
        components = best_state.components
        assert any([
            components.get("role"),
            components.get("steps"),
            components.get("output_format")
        ])
    
    def test_mutation_operations(self):
        """
        Test mutation operations.
        """
        # Apply mutation to a state
        mutated_state = self.mutator.mutate(self.structured_state)
        
        # Mutated state should be different but have similar structure
        assert mutated_state.text != self.structured_state.text
        
        # Some components should be preserved
        original_components = self.structured_state.components
        mutated_components = mutated_state.components
        
        # At least some components should be the same
        common_keys = set(original_components.keys()) & set(mutated_components.keys())
        assert len(common_keys) > 0
        
        # Test different mutation strengths
        self.mutator.set_mutation_strength(0.2)
        mild_mutation = self.mutator.mutate(self.structured_state)
        
        self.mutator.set_mutation_strength(0.8)
        strong_mutation = self.mutator.mutate(self.structured_state)
        
        # Should be able to apply multiple mutations
        multi_mutated = self.mutator.mutate(mild_mutation)
        assert multi_mutated.text != mild_mutation.text
    
    def test_crossover_operations(self):
        """
        Test crossover operations.
        """
        # Create a second parent state that's different from the structured state
        parent2_text = """
        Role: Text Analysis Specialist
        Task: Analyze the text and provide insights.
        
        Steps:
        - Read the text thoroughly
        - Identify key themes and patterns
        - Summarize main ideas
        - Provide analysis of themes
        
        Output Format: Provide a structured analysis with themes and evidence.
        """
        parent2 = PromptState(parent2_text)
        
        # Apply crossover
        child_state = self.crossover.crossover(self.structured_state, parent2)
        
        # Child should be different from both parents
        assert child_state.text != self.structured_state.text
        assert child_state.text != parent2.text
        
        # Child should have components from both parents
        child_components = child_state.components
        parent1_components = self.structured_state.components
        parent2_components = parent2.components
        
        # Should have at least one component from each parent
        inherited_from_p1 = False
        inherited_from_p2 = False
        
        for key in child_components:
            if key in parent1_components and child_components[key] == parent1_components[key]:
                inherited_from_p1 = True
            if key in parent2_components and child_components[key] == parent2_components[key]:
                inherited_from_p2 = True
        
        assert inherited_from_p1
        assert inherited_from_p2
    
    def test_evolution_selection(self):
        """
        Test evolution selection mechanisms.
        """
        # Create some nodes with different rewards
        nodes = []
        for i in range(5):
            node = MCTSNode(state=self.basic_state)
            # Set different rewards
            reward = 0.2 + (i * 0.2)  # 0.2, 0.4, 0.6, 0.8, 1.0
            node.update_statistics(reward)
            nodes.append(node)
        
        # Test tournament selection
        tournament_selected = self.evo_selector.tournament_select(nodes, tournament_size=3)
        assert tournament_selected in nodes
        
        # Test roulette wheel selection
        roulette_selected = self.evo_selector.roulette_wheel_select(nodes)
        assert roulette_selected in nodes
        
        # Test rank selection
        rank_selected = self.evo_selector.rank_select(nodes)
        assert rank_selected in nodes
        
        # Higher selection pressure should favor higher rewards
        self.evo_selector.set_selection_pressure(1.0)
        high_pressure_selections = [self.evo_selector.roulette_wheel_select(nodes) for _ in range(10)]
        high_avg_reward = sum(node.avg_reward for node in high_pressure_selections) / 10
        
        self.evo_selector.set_selection_pressure(0.1)
        low_pressure_selections = [self.evo_selector.roulette_wheel_select(nodes) for _ in range(10)]
        low_avg_reward = sum(node.avg_reward for node in low_pressure_selections) / 10
        
        # Higher pressure should generally select higher rewards
        # This is probabilistic, so could occasionally fail
        assert high_avg_reward >= low_avg_reward
        
        # Test diverse pair selection
        parent1, parent2 = self.evo_selector.select_diverse_pair(nodes)
        assert parent1 in nodes
        assert parent2 in nodes
        assert parent1 != parent2
    
    def test_error_collection_and_analysis(self):
        """
        Test error collection and analysis functionality.
        """
        # Create mock examples
        examples = [
            {"text": "I love this product!", "expected": "positive"},
            {"text": "This is terrible service.", "expected": "negative"},
            {"text": "The item arrived on time.", "expected": "neutral"},
            {"text": "I'm very disappointed.", "expected": "negative"},
            {"text": "Amazing experience!", "expected": "positive"}
        ]
        
        # Collect errors (will generate mock errors since no LLM is provided)
        errors = self.error_collector.collect_errors(self.basic_state, examples)
        
        # Should have some errors
        assert len(errors) > 0
        
        # Each error should have expected fields
        for error in errors:
            assert "example_id" in error
            assert "example" in error
            assert "error_type" in error
        
        # Analyze errors
        analysis = self.error_analyzer.analyze_errors(errors)
        
        # Analysis should have expected structure
        assert "error_clusters" in analysis
        assert "patterns" in analysis
        assert "summary" in analysis
        
        # Should have at least one error cluster
        assert len(analysis["error_clusters"]) > 0
        
        # Generate feedback from analysis
        feedback = self.feedback_generator.generate_feedback(analysis)
        
        # Should have some feedback items
        assert len(feedback) > 0
        
        # Each feedback item should have expected fields
        for item in feedback:
            assert "type" in item
            assert "description" in item
            assert "suggestion" in item
            assert "impact" in item
            assert "action_mapping" in item
        
        # Map feedback to actions
        actions = self.feedback_generator.map_feedback_to_actions(feedback)
        
        # Should have some actions
        assert len(actions) > 0
        
        # Actions should be applicable to the state
        for action in actions:
            assert action.is_applicable(self.basic_state)
    
    def test_integrated_error_feedback_in_mcts(self):
        """
        Test integration of error feedback in MCTS.
        """
        # Create a simplified MCTS engine with error feedback configuration
        mcts_with_feedback = MCTSEngine(
            transition=self.transition,
            reward_function=self.reward_fn,
            max_iterations=10,
            time_limit=3.0,
            evolution_config={
                "mutation_rate": 0.1,
                "crossover_rate": 0.1,
                "error_feedback_rate": 0.8  # High error feedback rate
            }
        )
        
        # Run optimization with error feedback focus
        best_state, stats = mcts_with_feedback.optimize(self.basic_state)
        
        # Should have performed some error feedback actions
        assert stats["error_feedback_actions"] > 0
        
        # Best state should be better than initial state
        initial_reward = self.reward_fn.calculate(self.basic_state)
        best_reward = self.reward_fn.calculate(best_state)
        assert best_reward > initial_reward
    
    def test_dynamic_operation_balance(self):
        """
        Test dynamic balancing of operations in MCTS.
        """
        # Create an MCTS engine with dynamic adjustment
        dynamic_mcts = MCTSEngine(
            transition=self.transition,
            reward_function=self.reward_fn,
            max_iterations=30,  # More iterations to see adjustment
            time_limit=5.0,
            evolution_config={
                "mutation_rate": 0.2,
                "crossover_rate": 0.2,
                "error_feedback_rate": 0.6,
                "adaptive_adjustment": True
            }
        )
        
        # Capture the initial rates
        initial_mutation = dynamic_mcts.evolution_config["mutation_rate"]
        initial_crossover = dynamic_mcts.evolution_config["crossover_rate"]
        initial_feedback = dynamic_mcts.evolution_config["error_feedback_rate"]
        
        # Run optimization
        best_state, stats = dynamic_mcts.optimize(self.basic_state)
        
        # Rates should have changed during optimization
        assert dynamic_mcts.evolution_config["mutation_rate"] != initial_mutation or \
               dynamic_mcts.evolution_config["crossover_rate"] != initial_crossover or \
               dynamic_mcts.evolution_config["error_feedback_rate"] != initial_feedback
        
        # Best state should be better than initial state
        initial_reward = self.reward_fn.calculate(self.basic_state)
        best_reward = self.reward_fn.calculate(best_state)
        assert best_reward > initial_reward