"""
Optimal prompt selection strategies for the final output generation.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from app.core.mcts.node import MCTSNode
from app.core.mdp.state import PromptState
from app.core.mdp.reward import RewardFunction
from app.utils.logger import get_logger

logger = get_logger("optimization.prompt_selector")

class PromptSelector:
    """
    Handles the selection of optimal prompts from the MCTS search tree.
    
    This class implements various selection strategies for identifying the best
    prompts based on path evaluation, trajectory analysis, and composite scoring.
    """
    
    def __init__(self, reward_function: Optional[RewardFunction] = None):
        """
        Initialize a prompt selector.
        
        Args:
            reward_function: Optional reward function for evaluating prompts.
        """
        self.reward_function = reward_function
        logger.debug("Initialized PromptSelector")
    
    def select_optimal_prompt(self, root_node: MCTSNode, strategy: str = "composite") -> Tuple[PromptState, Dict[str, Any]]:
        """
        Select the optimal prompt from the MCTS search tree.
        
        Args:
            root_node: Root node of the MCTS search tree.
            strategy: Selection strategy ('global_max', 'path_max', 'composite', 'stable').
            
        Returns:
            Tuple of (optimal_state, selection_stats).
        """
        logger.info(f"Selecting optimal prompt using {strategy} strategy")
        
        if strategy == "global_max":
            return self._select_global_max(root_node)
        elif strategy == "path_max":
            return self._select_path_max(root_node)
        elif strategy == "stable":
            return self._select_most_stable(root_node)
        else:  # default to composite
            return self._select_composite(root_node)
    
    def _select_global_max(self, root_node: MCTSNode) -> Tuple[PromptState, Dict[str, Any]]:
        """
        Select the node with the highest reward across the entire tree.
        
        Args:
            root_node: Root node of the MCTS search tree.
            
        Returns:
            Tuple of (optimal_state, selection_stats).
        """
        # Collect all nodes
        all_nodes = []
        self._collect_all_nodes(root_node, all_nodes)
        
        # Find node with highest reward
        best_node = max(all_nodes, key=lambda node: node.avg_reward if node.visit_count > 0 else -float('inf'))
        
        stats = {
            "strategy": "global_max",
            "nodes_evaluated": len(all_nodes),
            "best_reward": best_node.avg_reward,
            "best_node_depth": self._get_node_depth(best_node),
            "best_node_visits": best_node.visit_count
        }
        
        logger.debug(f"Selected global max node with reward {best_node.avg_reward:.4f}")
        return best_node.state, stats
    
    def _select_path_max(self, root_node: MCTSNode) -> Tuple[PromptState, Dict[str, Any]]:
        """
        Select the highest-reward node on the most promising path.
        
        Args:
            root_node: Root node of the MCTS search tree.
            
        Returns:
            Tuple of (optimal_state, selection_stats).
        """
        # Find the most visited child from root
        if not root_node.children:
            return root_node.state, {"strategy": "path_max", "best_reward": root_node.avg_reward}
        
        current = root_node
        path = [current]
        
        # Follow the most visited path
        while current.children:
            # Find child with most visits
            most_visited = max(current.children.values(), key=lambda node: node.visit_count)
            current = most_visited
            path.append(current)
        
        # Find the highest reward node along this path
        best_node = max(path, key=lambda node: node.avg_reward if node.visit_count > 0 else -float('inf'))
        
        stats = {
            "strategy": "path_max",
            "path_length": len(path),
            "best_reward": best_node.avg_reward,
            "best_node_depth": self._get_node_depth(best_node),
            "best_node_index": path.index(best_node)
        }
        
        logger.debug(f"Selected path max node with reward {best_node.avg_reward:.4f} at depth {stats['best_node_depth']}")
        return best_node.state, stats
    
    def _select_composite(self, root_node: MCTSNode) -> Tuple[PromptState, Dict[str, Any]]:
        """
        Select using a composite score considering reward, depth, and visit stability.
        
        Args:
            root_node: Root node of the MCTS search tree.
            
        Returns:
            Tuple of (optimal_state, selection_stats).
        """
        # Collect all nodes
        all_nodes = []
        self._collect_all_nodes(root_node, all_nodes)
        
        # Filter nodes with sufficient visits
        min_visits = max(5, root_node.visit_count // 20)  # At least 5% of root visits
        qualified_nodes = [node for node in all_nodes if node.visit_count >= min_visits]
        
        if not qualified_nodes:
            # Fall back to global max if no qualified nodes
            return self._select_global_max(root_node)
        
        # Calculate composite scores
        best_node = None
        best_score = -float('inf')
        scores = {}
        
        for node in qualified_nodes:
            depth = self._get_node_depth(node)
            visit_ratio = node.visit_count / root_node.visit_count
            
            # Composite score formula: reward + depth_bonus + visit_stability
            reward_component = node.avg_reward * 0.7  # 70% weight to reward
            depth_bonus = min(0.15, 0.03 * depth)  # Up to 15% for depth (encouraging deeper solutions)
            visit_stability = 0.15 * visit_ratio  # Up to 15% for visit stability
            
            composite_score = reward_component + depth_bonus + visit_stability
            scores[node] = {
                "composite_score": composite_score,
                "reward_component": reward_component,
                "depth_bonus": depth_bonus,
                "visit_stability": visit_stability,
                "depth": depth,
                "visits": node.visit_count
            }
            
            if composite_score > best_score:
                best_score = composite_score
                best_node = node
        
        if best_node is None:
            best_node = root_node
            
        stats = {
            "strategy": "composite",
            "nodes_evaluated": len(all_nodes),
            "qualified_nodes": len(qualified_nodes),
            "best_composite_score": best_score,
            "best_reward": best_node.avg_reward,
            "best_node_depth": self._get_node_depth(best_node),
            "best_node_visits": best_node.visit_count,
            "score_breakdown": scores.get(best_node, {})
        }
        
        logger.debug(f"Selected composite node with score {best_score:.4f}, reward {best_node.avg_reward:.4f}")
        return best_node.state, stats
    
    def _select_most_stable(self, root_node: MCTSNode) -> Tuple[PromptState, Dict[str, Any]]:
        """
        Select the most stable high-performing node based on visit count and reward consistency.
        
        Args:
            root_node: Root node of the MCTS search tree.
            
        Returns:
            Tuple of (optimal_state, selection_stats).
        """
        # Collect nodes with reasonable number of visits
        min_visits = max(10, root_node.visit_count // 10)  # At least 10% of root visits
        
        all_nodes = []
        self._collect_all_nodes(root_node, all_nodes)
        qualified_nodes = [node for node in all_nodes if node.visit_count >= min_visits]
        
        if not qualified_nodes:
            # Fall back to global max if no qualified nodes
            return self._select_global_max(root_node)
        
        # Find nodes with reward within 95% of the best reward
        max_reward = max(node.avg_reward for node in qualified_nodes)
        threshold = max_reward * 0.95
        stable_candidates = [node for node in qualified_nodes if node.avg_reward >= threshold]
        
        # Select the most visited node among the high reward candidates
        best_node = max(stable_candidates, key=lambda node: node.visit_count)
        
        stats = {
            "strategy": "stable",
            "nodes_evaluated": len(all_nodes),
            "qualified_nodes": len(qualified_nodes),
            "stable_candidates": len(stable_candidates),
            "best_reward": best_node.avg_reward,
            "max_possible_reward": max_reward,
            "best_node_depth": self._get_node_depth(best_node),
            "best_node_visits": best_node.visit_count
        }
        
        logger.debug(f"Selected stable node with reward {best_node.avg_reward:.4f} and {best_node.visit_count} visits")
        return best_node.state, stats
    
    def evaluate_path(self, path: List[MCTSNode]) -> Dict[str, Any]:
        """
        Evaluate the quality of a path in the search tree.
        
        Args:
            path: List of nodes from root to leaf.
            
        Returns:
            Dictionary with path evaluation metrics.
        """
        if not path:
            return {"valid": False, "reason": "Empty path"}
        
        # Calculate average and max reward along the path
        rewards = [node.avg_reward for node in path if node.visit_count > 0]
        if not rewards:
            return {"valid": False, "reason": "No nodes with visits"}
        
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        
        # Calculate reward stability (standard deviation)
        reward_stability = np.std(rewards) if len(rewards) > 1 else 0
        
        # Calculate visit stability (how consistently the path was explored)
        visits = [node.visit_count for node in path]
        visit_stability = np.std(visits) / max(visits) if max(visits) > 0 else 1
        
        return {
            "valid": True,
            "path_length": len(path),
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "reward_stability": reward_stability,
            "visit_stability": visit_stability,
            "total_visits": sum(visits),
            "path_score": avg_reward * (1 - min(0.5, reward_stability)) * (1 - min(0.5, visit_stability))
        }
    
    def analyze_trajectories(self, root_node: MCTSNode, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Analyze the top-k trajectories in the search tree.
        
        Args:
            root_node: Root node of the MCTS search tree.
            top_k: Number of top trajectories to analyze.
            
        Returns:
            List of trajectory analysis results.
        """
        # Collect all leaf nodes
        leaf_nodes = []
        self._collect_leaf_nodes(root_node, leaf_nodes)
        
        # Get paths from root to each leaf
        trajectories = []
        for leaf in leaf_nodes:
            path = leaf.get_path_from_root()
            evaluation = self.evaluate_path(path)
            if evaluation["valid"]:
                trajectories.append({
                    "path": path,
                    "evaluation": evaluation,
                    "leaf_node": leaf
                })
        
        # Sort trajectories by path score
        trajectories.sort(key=lambda t: t["evaluation"]["path_score"], reverse=True)
        
        # Return top-k trajectories
        return trajectories[:top_k]
    
    def _collect_all_nodes(self, node: MCTSNode, result: List[MCTSNode]) -> None:
        """
        Collect all nodes in the tree.
        
        Args:
            node: Current node.
            result: List to collect nodes in.
        """
        result.append(node)
        for child in node.children.values():
            self._collect_all_nodes(child, result)
    
    def _collect_leaf_nodes(self, node: MCTSNode, result: List[MCTSNode]) -> None:
        """
        Collect all leaf nodes in the tree.
        
        Args:
            node: Current node.
            result: List to collect leaf nodes in.
        """
        if not node.children:
            result.append(node)
        else:
            for child in node.children.values():
                self._collect_leaf_nodes(child, result)
    
    def _get_node_depth(self, node: MCTSNode) -> int:
        """
        Calculate the depth of a node in the tree.
        
        Args:
            node: Node to calculate depth for.
            
        Returns:
            Depth of the node.
        """
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth