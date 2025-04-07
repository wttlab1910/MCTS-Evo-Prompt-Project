"""
Backpropagation for the Monte Carlo Tree Search (MCTS) algorithm.
"""
from typing import Dict, Any, List, Optional, Tuple
from app.core.mcts.node import MCTSNode
from app.utils.logger import get_logger

logger = get_logger("mcts.backprop")

class Backpropagator:
    """
    Handles backpropagation of rewards in the MCTS algorithm.
    """
    
    def __init__(self, discount_factor: float = 1.0):
        """
        Initialize a backpropagator.
        
        Args:
            discount_factor: Factor to discount rewards at each level (default: 1.0).
        """
        self.discount_factor = discount_factor
        logger.debug(f"Initialized Backpropagator with discount_factor={discount_factor}")
    
    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate a reward from a node to the root.
        
        Args:
            node: Starting node for backpropagation.
            reward: Reward value to backpropagate.
        """
        logger.debug(f"Starting backpropagation from node {node.node_id[:8]} with reward {reward:.4f}")
        
        current = node
        current_reward = reward
        
        while current is not None:
            # Update the statistics of the current node
            current.update_statistics(current_reward)
            
            # Apply discount factor for the next level
            current_reward *= self.discount_factor
            
            # Move to the parent
            current = current.parent
        
        logger.debug(f"Completed backpropagation from node {node.node_id[:8]}")
    
    def backpropagate_with_path(self, path: List[MCTSNode], reward: float) -> None:
        """
        Backpropagate a reward along a specified path.
        
        Args:
            path: List of nodes from leaf to root.
            reward: Reward value to backpropagate.
        """
        if not path:
            logger.warning("Attempted to backpropagate with empty path")
            return
        
        logger.debug(f"Starting backpropagation along path of length {len(path)} "
                     f"from node {path[0].node_id[:8]} with reward {reward:.4f}")
        
        current_reward = reward
        
        for node in path:
            # Update the statistics of the current node
            node.update_statistics(current_reward)
            
            # Apply discount factor for the next level
            current_reward *= self.discount_factor
        
        logger.debug(f"Completed backpropagation along path of length {len(path)}")
    
    def set_discount_factor(self, discount_factor: float) -> None:
        """
        Set the discount factor for backpropagation.
        
        Args:
            discount_factor: New discount factor.
        """
        self.discount_factor = discount_factor
        logger.debug(f"Set discount factor to {discount_factor}")