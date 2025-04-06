"""
Comprehensive tests for Phase 2: MDP Framework Construction.
"""
import pytest
from app.core.mdp.state import PromptState
from app.core.mdp.action import (
    Action, StructuralAction, ContentAction, 
    KnowledgeAction, FormatAction,
    AddRoleAction, AddGoalAction, ModifyWorkflowAction,
    AddConstraintAction, AddExplanationAction, AddExampleAction,
    AdjustDetailAction, AddDomainKnowledgeAction,
    ClarifyTerminologyAction, AddRuleAction,
    SpecifyFormatAction, AddTemplateAction, StructureOutputAction,
    create_action
)
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction, PerformanceEvaluator

class TestPhase2:
    """
    Tests for Phase 2 components.
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
        self.transition = StateTransition(stochasticity=0.0)
        
        # Create reward function
        self.reward_fn = RewardFunction()
    
    def test_end_to_end_mdp_framework(self):
        """
        Test end-to-end MDP framework workflow.
        """
        # Start with a basic state
        state = self.basic_state
        
        # Apply a sequence of actions to build a prompt
        actions = [
            AddRoleAction(parameters={"role_text": "Sentiment Analysis Expert"}),
            ModifyWorkflowAction(parameters={
                "steps": [
                    "Read the text carefully",
                    "Identify sentiment-bearing words and phrases",
                    "Determine overall sentiment"
                ]
            }),
            AddDomainKnowledgeAction(parameters={
                "knowledge_text": "Sentiment is typically classified as positive, negative, or neutral.",
                "domain": "sentiment_analysis"
            }),
            AddExampleAction(parameters={
                "example_text": "Input: 'I absolutely love this product!'\nOutput: Positive",
                "example_type": "input_output"
            }),
            SpecifyFormatAction(parameters={
                "format_text": "Provide the sentiment as 'Sentiment: [positive/negative/neutral]' followed by confidence level."
            })
        ]
        
        # Track states and rewards
        states = [state]
        rewards = [self.reward_fn.calculate(state)]
        
        for action in actions:
            # Apply action through transition function
            new_state = self.transition.apply(state, action)
            
            # Calculate reward
            reward = self.reward_fn.calculate(new_state)
            
            # Store state and reward
            states.append(new_state)
            rewards.append(reward)
            
            # Update current state
            state = new_state
        
        # Verify that the final state has improved from the initial state
        # Not all actions increase reward, so we check overall improvement instead
        assert rewards[-1] > rewards[0]
        
        # Check final state properties
        final_state = states[-1]
        
        # Verify role was added
        assert "Role:" in final_state.text
        assert "Sentiment Analysis Expert" in final_state.text
        
        # Verify steps were added
        assert "Steps:" in final_state.text
        assert "Read the text carefully" in final_state.text
        
        # Verify domain knowledge was added
        assert "Domain Knowledge" in final_state.text
        
        # Verify example was added
        assert "Example:" in final_state.text
        assert "I absolutely love this product" in final_state.text
        
        # Verify output format was added
        assert "Output Format:" in final_state.text
        assert "Sentiment: [positive/negative/neutral]" in final_state.text
    
    def test_state_representation_functionality(self):
        """
        Test state representation functionality.
        """
        # Test component extraction
        components = self.structured_state.components
        assert "role" in components
        assert "task" in components
        assert "steps" in components
        assert "output_format" in components
        
        # Test structural completeness calculation
        empty_completeness = self.empty_state.get_structural_completeness()
        basic_completeness = self.basic_state.get_structural_completeness()
        structured_completeness = self.structured_state.get_structural_completeness()
        
        assert empty_completeness < basic_completeness < structured_completeness
        assert structured_completeness > 0.8  # Should be high for a well-structured prompt
        
        # Test state comparison
        state1 = PromptState("This is a test.")
        state2 = PromptState("This is a test.")
        state3 = PromptState("This is different.")
        
        assert state1 == state2
        assert state1 != state3
        assert hash(state1) == hash(state2)
        assert hash(state1) != hash(state3)
    
    def test_action_space_functionality(self):
        """
        Test action space functionality.
        """
        # Test basic action functionality
        action = AddRoleAction(parameters={"role_text": "Test Role"})
        assert action.action_type == "structural.add_role"
        assert action.parameters["role_text"] == "Test Role"
        
        # Test action applicability
        role_action = AddRoleAction(parameters={"role_text": "New Role"})
        assert role_action.is_applicable(self.empty_state) == True
        
        # After applying, it should not be applicable again with replace=True
        new_state = role_action.apply(self.empty_state)
        assert role_action.is_applicable(new_state) == False
        
        # But can be applicable if replace=False
        append_role_action = AddRoleAction(parameters={"role_text": "Addition", "replace": False})
        assert append_role_action.is_applicable(new_state) == True
        
        # Test action effect on state
        format_action = SpecifyFormatAction(parameters={
            "format_text": "Provide output in JSON format."
        })
        
        before_components = self.basic_state.components.copy()
        after_state = format_action.apply(self.basic_state)
        after_components = after_state.components
        
        # Verify action changed the state
        assert "output_format" not in before_components or before_components["output_format"] != after_components["output_format"]
        assert "Output Format:" in after_state.text
        assert "JSON" in after_state.text
        
        # Test action creation utility
        created_action = create_action("add_example", parameters={
            "example_text": "This is an example.",
            "example_type": "input_output"
        })
        
        assert isinstance(created_action, AddExampleAction)
        assert created_action.parameters["example_text"] == "This is an example."
    
    def test_transition_function_functionality(self):
        """
        Test transition function functionality.
        """
        # Test deterministic transition
        action = AddRoleAction(parameters={"role_text": "Test Role"})
        det_transition = StateTransition(stochasticity=0.0)
        
        new_state1 = det_transition.apply(self.empty_state, action)
        new_state2 = det_transition.apply(self.empty_state, action)
        
        # Deterministic transitions should yield identical states
        assert new_state1.text == new_state2.text
        
        # Test stochastic transition
        stoch_transition = StateTransition(stochasticity=1.0)
        
        # Apply same action multiple times
        states = [stoch_transition.apply(self.structured_state, action) for _ in range(5)]
        texts = [state.text for state in states]
        
        # With high stochasticity, we should get some variations
        unique_texts = set(texts)
        assert len(unique_texts) > 1
        
        # Test parent-child relationship in transitions
        action = ModifyWorkflowAction(parameters={
            "steps": ["Test step 1", "Test step 2"]
        })
        
        child_state = self.transition.apply(self.basic_state, action)
        
        assert child_state.parent == self.basic_state
        assert child_state.action_applied == str(action)
        assert len(child_state.history) == len(self.basic_state.history) + 1
    
    def test_reward_function_functionality(self):
        """
        Test reward function functionality.
        """
        # Test basic reward calculation
        empty_reward = self.reward_fn.calculate(self.empty_state)
        basic_reward = self.reward_fn.calculate(self.basic_state)
        structured_reward = self.reward_fn.calculate(self.structured_state)
        
        # More structured prompts should have higher rewards
        assert empty_reward < basic_reward < structured_reward
        
        # Test with custom performance function
        def custom_performance(state):
            return 0.9  # High fixed performance
            
        custom_reward_fn = RewardFunction(task_performance_fn=custom_performance)
        
        # Rewards should be higher with a high-performance function
        assert custom_reward_fn.calculate(self.empty_state) > self.reward_fn.calculate(self.empty_state)
        
        # Test with task-specific evaluator
        sentiment_evaluator = PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
        sentiment_reward_fn = RewardFunction(task_performance_fn=sentiment_evaluator)
        
        # Structured state is a good sentiment analysis prompt
        sentiment_reward = sentiment_reward_fn.calculate(self.structured_state)
        assert sentiment_reward > 0.7  # Should be high
        
        # Test component rewards
        components = self.reward_fn.get_component_rewards(self.structured_state)
        
        assert "performance" in components
        assert "structural_completeness" in components
        assert "token_efficiency" in components
        assert "total" in components
        
        # Total should match the calculated reward
        assert abs(components["total"] - self.reward_fn.calculate(self.structured_state)) < 1e-6
    
    def test_action_types_and_effects(self):
        """
        Test various action types and their effects.
        """
        # Test each action type
        
        # Structural actions
        role_action = AddRoleAction(parameters={"role_text": "Test Role"})
        role_state = role_action.apply(self.empty_state)
        assert "Role: Test Role" in role_state.text
        
        goal_action = AddGoalAction(parameters={"goal_text": "Test Task"})
        goal_state = goal_action.apply(self.empty_state)
        assert "Task: Test Task" in goal_state.text
        
        workflow_action = ModifyWorkflowAction(parameters={
            "steps": ["Step 1", "Step 2"]
        })
        workflow_state = workflow_action.apply(self.empty_state)
        assert "Steps:" in workflow_state.text
        assert "- Step 1" in workflow_state.text
        
        constraint_action = AddConstraintAction(parameters={
            "constraint_text": "Test constraint"
        })
        constraint_state = constraint_action.apply(self.empty_state)
        assert "Constraint" in constraint_state.text
        
        # Content actions
        explanation_action = AddExplanationAction(parameters={
            "explanation_text": "Test explanation",
            "target": "task"
        })
        explanation_state = explanation_action.apply(self.basic_state)
        assert "Test explanation" in explanation_state.text
        
        example_action = AddExampleAction(parameters={
            "example_text": "Test example",
            "example_type": "input_output"
        })
        example_state = example_action.apply(self.empty_state)
        assert "Example:" in example_state.text
        assert "Test example" in example_state.text
        
        # Knowledge actions
        knowledge_action = AddDomainKnowledgeAction(parameters={
            "knowledge_text": "Test knowledge",
            "domain": "test_domain"
        })
        knowledge_state = knowledge_action.apply(self.empty_state)
        assert "Domain Knowledge" in knowledge_state.text
        assert "test_domain" in knowledge_state.text
        
        term_action = ClarifyTerminologyAction(parameters={
            "term": "API",
            "definition": "Application Programming Interface"
        })
        term_state = term_action.apply(PromptState("This uses an API."))
        assert "API" in term_state.text
        assert "Application Programming Interface" in term_state.text
        
        # Format actions
        format_action = SpecifyFormatAction(parameters={
            "format_text": "Test format"
        })
        format_state = format_action.apply(self.empty_state)
        assert "Output Format:" in format_state.text
        assert "Test format" in format_state.text
    
    def test_action_composition(self):
        """
        Test composition of multiple actions.
        """
        # Create a sequence of actions
        actions = [
            AddRoleAction(parameters={"role_text": "Test Expert"}),
            AddGoalAction(parameters={"goal_text": "Perform test task"}),
            ModifyWorkflowAction(parameters={
                "steps": ["Step 1", "Step 2", "Step 3"]
            }),
            SpecifyFormatAction(parameters={
                "format_text": "Output in test format"
            })
        ]
        
        # Apply actions sequentially
        state = self.empty_state
        for action in actions:
            state = action.apply(state)
        
        # Verify final state contains all components
        assert "Role: Test Expert" in state.text
        assert "Task: Perform test task" in state.text
        assert "Steps:" in state.text
        assert "- Step 1" in state.text
        assert "- Step 2" in state.text
        assert "- Step 3" in state.text
        assert "Output Format:" in state.text
        assert "Output in test format" in state.text
        
        # Verify components are correctly extracted
        assert state.components["role"] == "Test Expert"
        assert state.components["task"] == "Perform test task"
        assert len(state.components["steps"]) == 3
        assert state.components["output_format"] == "Output in test format"
        
        # Verify structural completeness is high
        assert state.get_structural_completeness() > 0.8