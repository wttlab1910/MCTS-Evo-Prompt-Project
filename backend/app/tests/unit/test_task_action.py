"""
Comprehensive tests for task-specific action generators.
"""

"""
Comprehensive tests for task-specific action generators.
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pytest
from typing import List, Dict, Any
from app.actions.task_actions import (
    get_task_action_generator,
    generate_table_task_actions,
    generate_counting_task_actions,
    generate_sequence_task_actions,
    generate_causal_task_actions,
    generate_epistemic_task_actions,
    generate_geometric_task_actions
)
from app.core.mdp.state import PromptState
from app.core.mdp.action import Action

# 其余代码保持不变...

class TestTaskActions:
    """
    Tests for task-specific action generators.
    """
    
    def setup_method(self):
        """
        Set up test fixtures.
        """
        # Create sample prompt states for different tasks
        self.table_state = PromptState("Answer the following penguins_in_a_table question.")
        self.counting_state = PromptState("Answer the following object_counting question.")
        self.sequence_state = PromptState("Answer the following temporal_sequences question.")
        self.causal_state = PromptState("Answer the following causal_judgment question.")
        self.epistemic_state = PromptState("Answer the following epistemic question.")
        self.geometric_state = PromptState("Answer the following geometric_shapes question.")
        
    def test_get_task_action_generator(self):
        """
        Test retrieval of task-specific action generators.
        """
        # Test getting generators for supported tasks
        table_generator = get_task_action_generator("penguins_in_a_table")
        counting_generator = get_task_action_generator("object_counting")
        sequence_generator = get_task_action_generator("temporal_sequences")
        causal_generator = get_task_action_generator("causal_judgment")
        epistemic_generator = get_task_action_generator("epistemic")
        geometric_generator = get_task_action_generator("geometric_shapes")
        
        # Verify correct generators are returned
        assert table_generator == generate_table_task_actions
        assert counting_generator == generate_counting_task_actions
        assert sequence_generator == generate_sequence_task_actions
        assert causal_generator == generate_causal_task_actions
        assert epistemic_generator == generate_epistemic_task_actions
        assert geometric_generator == generate_geometric_task_actions
        
        # Test non-existent task type
        unknown_generator = get_task_action_generator("unknown_task_type")
        assert unknown_generator is None
    
    def test_table_task_actions(self):
        """
        Test table-specific action generation.
        """
        # Generate actions for a table task
        actions = generate_table_task_actions(self.table_state)
        
        # Verify action count and types
        assert len(actions) >= 5, "Should generate at least 5 table-specific actions"
        
        # Verify action content
        action_types = [a.action_type for a in actions]
        action_descriptions = [str(a) for a in actions]
        
        # Check for specific table-related actions
        assert any("domain_knowledge" in t for t in action_types), "Should include domain knowledge actions"
        assert any("workflow" in t for t in action_types), "Should include workflow actions"
        assert any("format" in t for t in action_types), "Should include format actions"
        
        # Check for table-specific content
        table_keywords = ["table", "column", "row", "data", "lookup", "calculation", "comparison"]
        assert any(any(kw in d.lower() for kw in table_keywords) for d in action_descriptions), \
            "Should include table-specific terminology"
        
        # Check for penguin-specific content
        assert any("penguin" in d.lower() for d in action_descriptions), \
            "Should include penguin-specific content for this task"
    
    def test_counting_task_actions(self):
        """
        Test counting-specific action generation.
        """
        # Generate actions for a counting task
        actions = generate_counting_task_actions(self.counting_state)
        
        # Verify action count
        assert len(actions) >= 3, "Should generate at least 3 counting-specific actions"
        
        # Verify action content
        action_descriptions = [str(a) for a in actions]
        
        # Check for counting-specific content
        counting_keywords = ["count", "object", "identify", "track", "systematic"]
        assert any(any(kw in d.lower() for kw in counting_keywords) for d in action_descriptions), \
            "Should include counting-specific terminology"
    
    def test_sequence_task_actions(self):
        """
        Test sequence-specific action generation.
        """
        # Generate actions for a sequence task
        actions = generate_sequence_task_actions(self.sequence_state)
        
        # Verify action count
        assert len(actions) >= 3, "Should generate at least 3 sequence-specific actions"
        
        # Verify action content
        action_descriptions = [str(a) for a in actions]
        
        # Check for sequence-specific content
        sequence_keywords = ["pattern", "sequence", "temporal", "next", "identify"]
        assert any(any(kw in d.lower() for kw in sequence_keywords) for d in action_descriptions), \
            "Should include sequence-specific terminology"
    
    def test_causal_task_actions(self):
        """
        Test causal judgment action generation.
        """
        # Generate actions for a causal judgment task
        actions = generate_causal_task_actions(self.causal_state)
        
        # Verify action count
        assert len(actions) >= 3, "Should generate at least 3 causal-specific actions"
        
        # Verify action content
        action_descriptions = [str(a) for a in actions]
        
        # Check for causal-specific content
        causal_keywords = ["cause", "effect", "causal", "correlation"]
        assert any(any(kw in d.lower() for kw in causal_keywords) for d in action_descriptions), \
            "Should include causal-specific terminology"
    
    def test_epistemic_task_actions(self):
        """
        Test epistemic reasoning action generation.
        """
        # Generate actions for an epistemic task
        actions = generate_epistemic_task_actions(self.epistemic_state)
        
        # Verify action count
        assert len(actions) >= 3, "Should generate at least 3 epistemic-specific actions"
        
        # Verify action content
        action_descriptions = [str(a) for a in actions]
        
        # Check for epistemic-specific content
        epistemic_keywords = ["knowledge", "belief", "reason", "certainty", "inference"]
        assert any(any(kw in d.lower() for kw in epistemic_keywords) for d in action_descriptions), \
            "Should include epistemic-specific terminology"
    
    def test_geometric_task_actions(self):
        """
        Test geometric reasoning action generation.
        """
        # Generate actions for a geometric task
        actions = generate_geometric_task_actions(self.geometric_state)
        
        # Verify action count
        assert len(actions) >= 3, "Should generate at least 3 geometric-specific actions"
        
        # Verify action content
        action_descriptions = [str(a) for a in actions]
        
        # Check for geometric-specific content
        geometric_keywords = ["shape", "geometric", "position", "orientation", "spatial"]
        assert any(any(kw in d.lower() for kw in geometric_keywords) for d in action_descriptions), \
            "Should include geometric-specific terminology"
    
    def test_action_applicability(self):
        """
        Test that generated actions are applicable to the given states.
        """
        # Test each task type
        task_pairs = [
            (self.table_state, generate_table_task_actions),
            (self.counting_state, generate_counting_task_actions),
            (self.sequence_state, generate_sequence_task_actions),
            (self.causal_state, generate_causal_task_actions),
            (self.epistemic_state, generate_epistemic_task_actions),
            (self.geometric_state, generate_geometric_task_actions)
        ]
        
        for state, generator in task_pairs:
            actions = generator(state)
            
            # Verify all actions are applicable to their state
            for action in actions:
                assert action.is_applicable(state), \
                    f"Action {action} should be applicable to its corresponding state"
    
    def test_action_diversity(self):
        """
        Test diversity of generated actions.
        """
        # Generate actions for multiple tasks
        table_actions = generate_table_task_actions(self.table_state)
        counting_actions = generate_counting_task_actions(self.counting_state)
        
        # Convert to sets of action descriptions for comparison
        table_descs = {str(a) for a in table_actions}
        counting_descs = {str(a) for a in counting_actions}
        
        # Verify minimal overlap between task types
        overlap = table_descs.intersection(counting_descs)
        assert len(overlap) < min(len(table_descs), len(counting_descs)) / 2, \
            "Different task types should have minimal action overlap"
    
    def test_integration_with_mcts(self):
        """
        Test integration with MCTS engine by simulating a small optimization.
        """
        from app.core.mdp.transition import StateTransition
        from app.core.mdp.reward import RewardFunction
        from app.core.mcts.engine import MCTSEngine
        
        # Create simple components for testing
        transition = StateTransition()
        
        # Simple reward function that rewards actions containing task-specific keywords
        def simple_reward(state: PromptState) -> float:
            text = state.text.lower()
            # Table-specific keywords
            keywords = ["table", "column", "row", "calculation", "penguin"]
            matches = sum(1 for kw in keywords if kw in text)
            return min(1.0, matches / len(keywords))
        
        reward_fn = RewardFunction(task_performance_fn=simple_reward)
        
        # Create MCTS engine with table task action generator
        engine = MCTSEngine(
            transition=transition,
            reward_function=reward_fn,
            max_iterations=5,  # Just a few iterations for testing
            time_limit=5.0,
            max_depth=2,
            action_generator=generate_table_task_actions
        )
        
        # Run a mini optimization
        result_state, stats = engine.optimize(self.table_state)
        
        # Verify that optimization produced a better state
        assert stats["tree_size"] > 1, "Should create multiple nodes in the tree"
        assert "table" in result_state.text.lower(), "Should incorporate table-specific content"