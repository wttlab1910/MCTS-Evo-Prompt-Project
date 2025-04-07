"""
Expansion strategies for the Monte Carlo Tree Search (MCTS) algorithm.
"""
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import random
from app.core.mcts.node import MCTSNode
from app.core.mdp.state import PromptState
from app.core.mdp.action import Action, create_action
from app.core.mdp.transition import StateTransition
from app.utils.logger import get_logger

logger = get_logger("mcts.expansion")

class ActionExpander:
    """
    Expansion strategy for MCTS that applies actions to expand nodes.
    """
    
    def __init__(
        self, 
        transition: StateTransition,
        max_children_per_expansion: int = 5,
        action_filter: Optional[callable] = None
    ):
        """
        Initialize an action expander.
        
        Args:
            transition: State transition function to apply actions.
            max_children_per_expansion: Maximum number of children to create per expansion.
            action_filter: Optional function to filter applicable actions.
        """
        self.transition = transition
        self.max_children_per_expansion = max_children_per_expansion
        self.action_filter = action_filter
        
        logger.debug(f"Initialized ActionExpander with max_children_per_expansion={max_children_per_expansion}")
    
    def expand(self, node: MCTSNode, available_actions: List[Action]) -> List[MCTSNode]:
        """
        Expand a node by applying available actions.
        
        Args:
            node: Node to expand.
            available_actions: List of actions that can be applied.
            
        Returns:
            List of newly created child nodes.
        """
        # 修复: 检查是否已经完全展开
        if node.fully_expanded and not node.available_actions:
            logger.warning(f"Attempting to expand a fully expanded node {node.node_id[:8]}")
            return []
        
        # Filter actions if a filter function is provided
        if self.action_filter:
            filtered_actions = [action for action in available_actions 
                               if self.action_filter(node.state, action)]
        else:
            filtered_actions = [action for action in available_actions 
                               if action.is_applicable(node.state)]
        
        # Update available actions in the node - 修复: 仅添加新的可用actions
        available_action_strs = set([str(action) for action in filtered_actions])
        # 移除已有的子节点
        available_action_strs -= set(node.children.keys())
        # 设置可用的actions
        node.available_actions = available_action_strs
        
        # 修复: 检查是否还有可用的actions
        if not node.available_actions:
            node.mark_fully_expanded()
            logger.debug(f"No available actions for node {node.node_id[:8]}, marked as fully expanded")
            return []
        
        # Select actions to apply (either all available or limited by max_children_per_expansion)
        actions_to_apply = []
        available_action_strs = list(node.available_actions)
        
        if len(available_action_strs) <= self.max_children_per_expansion:
            # Use all available actions
            actions_to_apply = [action for action in filtered_actions 
                               if str(action) in available_action_strs]
        else:
            # Randomly select max_children_per_expansion actions
            selected_action_strs = random.sample(available_action_strs, self.max_children_per_expansion)
            actions_to_apply = [action for action in filtered_actions 
                               if str(action) in selected_action_strs]
        
        logger.debug(f"Selected {len(actions_to_apply)} actions to apply to node {node.node_id[:8]}")
        
        # Apply selected actions and create child nodes
        children = []
        for action in actions_to_apply:
            child_state = self.transition.apply(node.state, action)
            child_node = node.add_child(action, child_state)
            children.append(child_node)
        
        # If all available actions have been applied, mark as fully expanded
        if not node.available_actions:
            node.mark_fully_expanded()
            logger.debug(f"All available actions applied to node {node.node_id[:8]}, marked as fully expanded")
        
        logger.debug(f"Created {len(children)} child nodes for node {node.node_id[:8]}")
        return children
    
    def expand_with_action(self, node: MCTSNode, action: Action) -> Optional[MCTSNode]:
        """
        Expand a node by applying a specific action.
        
        Args:
            node: Node to expand.
            action: Action to apply.
            
        Returns:
            Newly created child node or None if the action is not applicable.
        """
        action_str = str(action)
        
        # Check if the action has already been applied
        if action_str in node.children:
            logger.debug(f"Action {action_str} already applied to node {node.node_id[:8]}")
            return node.children[action_str]
        
        # Check if the action is applicable
        if not action.is_applicable(node.state):
            logger.debug(f"Action {action_str} not applicable to node {node.node_id[:8]}")
            return None
        
        # Apply the action and create a child node
        child_state = self.transition.apply(node.state, action)
        child_node = node.add_child(action, child_state)
        
        # Update available actions
        if action_str in node.available_actions:
            node.available_actions.remove(action_str)
        
        # If all available actions have been applied, mark as fully expanded
        if not node.available_actions:
            node.mark_fully_expanded()
            logger.debug(f"All available actions applied to node {node.node_id[:8]}, marked as fully expanded")
        
        logger.debug(f"Created child node {child_node.node_id[:8]} for node {node.node_id[:8]} via action {action_str}")
        return child_node

    def generate_actions_for_node(self, node: MCTSNode) -> List[Action]:
        """
        Generate a list of applicable actions for a node based on its state.
        
        Args:
            node: Node to generate actions for.
            
        Returns:
            List of applicable actions.
        """
        state = node.state
        actions = []
        
        # Check for missing or enhanceable components
        if not state.has_component("role"):
            actions.append(
                create_action("add_role", parameters={
                    "role_text": "Expert in the relevant domain"
                })
            )
        
        if not state.has_component("task"):
            actions.append(
                create_action("add_goal", parameters={
                    "goal_text": "Complete the assigned task effectively"
                })
            )
        
        if not state.has_component("steps"):
            actions.append(
                create_action("modify_workflow", parameters={
                    "steps": [
                        "Analyze the input thoroughly",
                        "Process the information systematically",
                        "Generate an appropriate response"
                    ]
                })
            )
        
        if not state.has_component("output_format"):
            actions.append(
                create_action("specify_format", parameters={
                    "format_text": "Provide a clear and structured response"
                })
            )
        
        if "examples" not in state.components:
            actions.append(
                create_action("add_example", parameters={
                    "example_text": "This is a sample example of expected input and output",
                    "example_type": "input_output"
                })
            )
        
        # Add other potential actions
        actions.extend([
            create_action("add_constraint", parameters={
                "constraint_text": "Ensure responses are concise and relevant"
            }),
            create_action("add_domain_knowledge", parameters={
                "knowledge_text": "Apply domain-specific knowledge where relevant",
                "domain": "general"
            }),
            create_action("add_explanation", parameters={
                "explanation_text": "Provide clear explanations for complex concepts",
                "target": "task"
            })
        ])
        
        # Filter out actions that are not applicable
        actions = [action for action in actions if action.is_applicable(state)]
        
        logger.debug(f"Generated {len(actions)} applicable actions for node {node.node_id[:8]}")
        return actions