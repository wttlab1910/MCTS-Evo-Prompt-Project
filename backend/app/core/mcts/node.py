"""
Node representation for the Monte Carlo Tree Search (MCTS) algorithm.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
import uuid
from app.core.mdp.state import PromptState
from app.core.mdp.action import Action
from app.utils.logger import get_logger

logger = get_logger("mcts.node")

class MCTSNode:
    """
    Node in the MCTS tree representing a prompt state.
    
    Each node contains:
    - A prompt state
    - Statistics (visit count, total reward, etc.)
    - Children nodes
    - Parent reference
    """
    
    def __init__(
        self, 
        state: PromptState,
        parent: Optional['MCTSNode'] = None,
        action_applied: Optional[Action] = None,
        node_id: Optional[str] = None
    ):
        """
        Initialize an MCTS node.
        
        Args:
            state: The prompt state associated with this node.
            parent: Parent node (None for root).
            action_applied: Action applied to reach this node.
            node_id: Optional ID for the node (generated if not provided).
        """
        self.state = state
        self.parent = parent
        self.action_applied = action_applied
        self.node_id = node_id or str(uuid.uuid4())
        
        # Statistics
        self.visit_count = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        
        # Children management
        self.children: Dict[str, 'MCTSNode'] = {}  # Map from action string to child node
        self.available_actions: Set[str] = set()  # Actions not yet expanded
        self.fully_expanded = False
        
        # Evolutionary information
        self.evolution_history: List[str] = []  # List of evolutionary operations applied
        self.generation = 0 if parent is None else parent.generation + 1
        
        # Error feedback
        self.error_feedback: List[Dict[str, Any]] = []
        
        logger.debug(f"Created node {self.node_id[:8]} with state {self.state}")
    
    def add_child(self, action: Action, child_state: PromptState) -> 'MCTSNode':
        """
        Add a child node to this node.
        
        Args:
            action: Action applied to reach the child.
            child_state: Resulting state after applying the action.
            
        Returns:
            The newly created child node.
        """
        action_str = str(action)
        child = MCTSNode(
            state=child_state,
            parent=self,
            action_applied=action
        )
        self.children[action_str] = child
        if action_str in self.available_actions:
            self.available_actions.remove(action_str)
        
        logger.debug(f"Added child {child.node_id[:8]} to node {self.node_id[:8]} via action {action_str}")
        return child
    
    def update_statistics(self, reward: float) -> None:
        """
        Update the statistics of this node based on a new reward.
        
        Args:
            reward: Reward value to incorporate.
        """
        self.visit_count += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.visit_count
        
        logger.debug(f"Updated node {self.node_id[:8]}: visits={self.visit_count}, "
                     f"avg_reward={self.avg_reward:.4f}")
    
    def set_available_actions(self, actions: List[Action]) -> None:
        """
        Set the available actions for this node.
        
        Args:
            actions: List of actions that can be applied to this node's state.
        """
        self.available_actions = {str(action) for action in actions}
        
        logger.debug(f"Set {len(self.available_actions)} available actions for node {self.node_id[:8]}")
    
    def mark_fully_expanded(self) -> None:
        """Mark this node as fully expanded."""
        self.fully_expanded = True
        logger.debug(f"Node {self.node_id[:8]} marked as fully expanded")
    
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf node (has no children).
        
        Returns:
            True if the node has no children, False otherwise.
        """
        return len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """
        Check if this node is fully expanded.
        
        Returns:
            True if all available actions have been explored, False otherwise.
        """
        # If marked as fully expanded or no more available actions
        return self.fully_expanded or len(self.available_actions) == 0
    
    def add_evolution_operation(self, operation: str) -> None:
        """
        Add an evolutionary operation to this node's history.
        
        Args:
            operation: Description of the evolutionary operation.
        """
        self.evolution_history.append(operation)
        logger.debug(f"Added evolution operation to node {self.node_id[:8]}: {operation}")
    
    def add_error_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Add error feedback to this node.
        
        Args:
            feedback: Dictionary containing error information and feedback.
        """
        self.error_feedback.append(feedback)
        logger.debug(f"Added error feedback to node {self.node_id[:8]}")
    
    def get_path_from_root(self) -> List['MCTSNode']:
        """
        Get the path from the root to this node.
        
        Returns:
            List of nodes from root to this node.
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path
    
    def get_action_path_from_root(self) -> List[Optional[Action]]:
        """
        Get the sequence of actions from the root to this node.
        
        Returns:
            List of actions from root to this node.
        """
        path = []
        current = self
        while current is not None:
            path.append(current.action_applied)
            current = current.parent
        path.reverse()
        # Remove the first element (None for root)
        if path and path[0] is None:
            path = path[1:]
        return path
    
    def __str__(self) -> str:
        """
        Get a string representation of this node.
        
        Returns:
            String representation.
        """
        return f"MCTSNode(id={self.node_id[:8]}, visits={self.visit_count}, " \
               f"avg_reward={self.avg_reward:.4f}, children={len(self.children)})"
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of this node.
        
        Returns:
            Detailed string representation.
        """
        return f"MCTSNode(id={self.node_id[:8]}, visits={self.visit_count}, " \
               f"avg_reward={self.avg_reward:.4f}, children={len(self.children)}, " \
               f"generation={self.generation})"