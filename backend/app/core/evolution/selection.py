"""
Selection mechanisms for evolutionary prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import random
import math
from app.core.mcts.node import MCTSNode
from app.utils.logger import get_logger

logger = get_logger("evolution.selection")

class EvolutionSelector:
    """
    Implements selection mechanisms for evolutionary operations.
    
    This class handles selecting promising prompts for mutation and crossover,
    using various strategies like tournament selection, roulette wheel, etc.
    """
    
    def __init__(self, selection_pressure: float = 0.7):
        """
        Initialize an evolution selector.
        
        Args:
            selection_pressure: Controls how strongly selection favors higher-quality prompts (0.0 to 1.0).
        """
        self.selection_pressure = min(1.0, max(0.0, selection_pressure))
        logger.debug(f"Initialized EvolutionSelector with selection_pressure={selection_pressure}")
    
    def tournament_select(self, nodes: List[MCTSNode], tournament_size: int = 3) -> MCTSNode:
        """
        Select a node using tournament selection.
        
        Args:
            nodes: List of nodes to select from.
            tournament_size: Number of nodes to include in each tournament.
            
        Returns:
            Selected node.
            
        Raises:
            ValueError: If the nodes list is empty.
        """
        if not nodes:
            raise ValueError("Cannot select from an empty list of nodes")
        
        if len(nodes) <= tournament_size:
            # If fewer nodes than tournament size, just pick the best
            return max(nodes, key=lambda n: n.avg_reward)
        
        # Randomly select tournament_size nodes for the tournament
        tournament = random.sample(nodes, tournament_size)
        
        # Select the node with the highest average reward
        best_node = max(tournament, key=lambda n: n.avg_reward)
        
        logger.debug(f"Tournament selected node {best_node.node_id[:8]} with "
                    f"avg_reward={best_node.avg_reward:.4f}")
        return best_node
    
    def roulette_wheel_select(self, nodes: List[MCTSNode]) -> MCTSNode:
        """
        Select a node using roulette wheel selection (fitness proportionate).
        
        Args:
            nodes: List of nodes to select from.
            
        Returns:
            Selected node.
            
        Raises:
            ValueError: If the nodes list is empty.
        """
        if not nodes:
            raise ValueError("Cannot select from an empty list of nodes")
        
        # Calculate fitness for each node (using average reward)
        # Add a small value to avoid issues with negative or zero rewards
        base_fitness = [max(0.01, n.avg_reward) for n in nodes]
        
        # Apply selection pressure
        if self.selection_pressure > 0:
            # Find the minimum and maximum fitness
            min_fitness = min(base_fitness)
            max_fitness = max(base_fitness)
            
            # Adjust fitness based on selection pressure
            if max_fitness > min_fitness:
                fitness = []
                for f in base_fitness:
                    # Linear scaling based on selection pressure
                    scaled = min_fitness + (f - min_fitness) * self.selection_pressure
                    fitness.append(scaled)
            else:
                fitness = base_fitness
        else:
            fitness = base_fitness
        
        # Calculate total fitness
        total_fitness = sum(fitness)
        
        if total_fitness <= 0:
            # If total fitness is zero or negative, select randomly
            selected = random.choice(nodes)
        else:
            # Normalize fitness
            normalized_fitness = [f / total_fitness for f in fitness]
            
            # Select based on normalized fitness
            selected = random.choices(nodes, weights=normalized_fitness, k=1)[0]
        
        logger.debug(f"Roulette wheel selected node {selected.node_id[:8]} with "
                    f"avg_reward={selected.avg_reward:.4f}")
        return selected
    
    def rank_select(self, nodes: List[MCTSNode]) -> MCTSNode:
        """
        Select a node using rank selection.
        
        Args:
            nodes: List of nodes to select from.
            
        Returns:
            Selected node.
            
        Raises:
            ValueError: If the nodes list is empty.
        """
        if not nodes:
            raise ValueError("Cannot select from an empty list of nodes")
        
        # Sort nodes by average reward
        sorted_nodes = sorted(nodes, key=lambda n: n.avg_reward)
        
        # Assign ranks (1 to N)
        ranks = list(range(1, len(sorted_nodes) + 1))
        
        # Apply selection pressure to ranks
        if self.selection_pressure > 0:
            # Higher selection pressure means more weight to higher ranks
            weights = [rank ** self.selection_pressure for rank in ranks]
        else:
            weights = ranks
        
        # Select based on weights
        selected = random.choices(sorted_nodes, weights=weights, k=1)[0]
        
        logger.debug(f"Rank selected node {selected.node_id[:8]} with "
                    f"avg_reward={selected.avg_reward:.4f}")
        return selected
    
    def select_diverse_pair(self, nodes: List[MCTSNode]) -> Tuple[MCTSNode, MCTSNode]:
        """
        Select a diverse pair of nodes for crossover.
        
        Args:
            nodes: List of nodes to select from.
            
        Returns:
            Tuple of (parent1, parent2).
            
        Raises:
            ValueError: If fewer than 2 nodes are provided.
        """
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes to select a pair")
        
        # Select first parent using tournament selection
        parent1 = self.tournament_select(nodes)
        
        # For second parent, try to find one that is different from the first
        # First, remove parent1 from consideration
        remaining = [n for n in nodes if n != parent1]
        
        if not remaining:
            # This shouldn't happen given the check above, but just in case
            return parent1, random.choice(nodes)
        
        # Calculate dissimilarity scores for remaining nodes
        # For now, use a simple heuristic based on difference in components
        scores = []
        for node in remaining:
            # Count components that differ between the two states
            diff_count = 0
            all_components = set(parent1.state.components.keys()) | set(node.state.components.keys())
            
            for component in all_components:
                in_parent1 = component in parent1.state.components
                in_node = component in node.state.components
                
                if in_parent1 != in_node:
                    # One has it, one doesn't
                    diff_count += 1
                elif in_parent1 and in_node:
                    # Both have it, but compare values
                    if parent1.state.components[component] != node.state.components[component]:
                        diff_count += 1
            
            # Weight by both dissimilarity and quality
            quality_score = node.avg_reward
            dissimilarity_score = diff_count / max(1, len(all_components))
            
            # Combine scores (weighted sum)
            combined_score = 0.7 * quality_score + 0.3 * dissimilarity_score
            scores.append(combined_score)
        
        # Select based on combined scores
        if sum(scores) <= 0:
            parent2 = random.choice(remaining)
        else:
            # Normalize scores
            total = sum(scores)
            normalized = [s / total for s in scores]
            parent2 = random.choices(remaining, weights=normalized, k=1)[0]
        
        logger.debug(f"Selected diverse pair: {parent1.node_id[:8]} and {parent2.node_id[:8]}")
        return parent1, parent2
    
    def set_selection_pressure(self, pressure: float) -> None:
        """
        Set the selection pressure.
        
        Args:
            pressure: New selection pressure (0.0 to 1.0).
        """
        self.selection_pressure = min(1.0, max(0.0, pressure))
        logger.debug(f"Set selection pressure to {self.selection_pressure}")