"""
Selection strategies for the Monte Carlo Tree Search (MCTS) algorithm.
"""
from typing import Dict, Any, List, Optional, Tuple
import math
from app.core.mcts.node import MCTSNode
from app.utils.logger import get_logger

logger = get_logger("mcts.selection")

class UCTSelector:
    """
    UCT (Upper Confidence Bound for Trees) selection strategy for MCTS.
    
    This strategy balances exploration and exploitation by selecting nodes
    based on their average reward and visit count.
    """
    
    def __init__(self, exploration_weight: float = 1.41):
        """
        Initialize a UCT selector.
        
        Args:
            exploration_weight: Weight for the exploration term in UCT formula.
                Default is sqrt(2) which is a common value.
        """
        self.exploration_weight = exploration_weight
        logger.debug(f"Initialized UCTSelector with exploration_weight={exploration_weight}")
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to explore using UCT.
        
        Starting from the given node, traverse the tree until reaching a node
        that is not fully expanded or is a leaf.
        
        Args:
            node: Starting node for selection.
            
        Returns:
            Selected node for expansion.
        """
        logger.debug(f"Starting selection from node {node.node_id[:8]}")
        
        # Traverse the tree until reaching a node that is not fully expanded or is a leaf
        current = node
        while not current.is_leaf() and current.is_fully_expanded():
            current = self._select_best_child(current)
        
        logger.debug(f"Selected node {current.node_id[:8]} for expansion")
        return current
    
    def _select_best_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select the best child of a node using UCT formula.
        
        Args:
            node: Parent node.
            
        Returns:
            Best child node according to UCT.
            
        Raises:
            ValueError: If the node has no children.
        """
        if node.is_leaf():
            raise ValueError("Cannot select best child of a leaf node")
        
        best_score = float('-inf')
        best_child = None
        
        for child in node.children.values():
            # UCT formula: avg_reward + exploration_weight * sqrt(ln(parent_visits) / child_visits)
            if child.visit_count == 0:
                # If child has not been visited, prioritize it
                score = float('inf')
            else:
                # Balance exploitation (avg_reward) and exploration (sqrt(ln(parent_visits) / child_visits))
                exploitation = child.avg_reward
                exploration = self.exploration_weight * math.sqrt(
                    math.log(node.visit_count) / child.visit_count)
                score = exploitation + exploration
            
            # Update best child if this one has a higher score
            if best_child is None or score > best_score:
                best_score = score
                best_child = child
        
        logger.debug(f"Selected best child {best_child.node_id[:8]} with UCT score {best_score:.4f}")
        return best_child
    
    def get_children_scores(self, node: MCTSNode) -> Dict[str, float]:
        """
        Calculate UCT scores for all children of a node.
        
        Args:
            node: Parent node.
            
        Returns:
            Dictionary mapping child node IDs to their UCT scores.
        """
        scores = {}
        
        for child_id, child in node.children.items():
            if child.visit_count == 0:
                scores[child_id] = float('inf')
            else:
                exploitation = child.avg_reward
                exploration = self.exploration_weight * math.sqrt(
                    math.log(node.visit_count) / child.visit_count)
                scores[child_id] = exploitation + exploration
        
        return scores
    
    def set_exploration_weight(self, weight: float) -> None:
        """
        Set the exploration weight for the UCT formula.
        
        Args:
            weight: New exploration weight.
        """
        self.exploration_weight = weight
        logger.debug(f"Set exploration weight to {weight}")