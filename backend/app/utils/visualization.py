"""
Visualization utilities for prompt optimization results.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from app.core.mcts.node import MCTSNode
from app.utils.logger import get_logger

logger = get_logger("utils.visualization")

class OptimizationVisualizer:
    """
    Visualization utilities for prompt optimization results.
    
    This class provides methods for generating visualizations of the
    optimization process, search tree, and final results.
    """
    
    @staticmethod
    def generate_tree_visualization(root_node: MCTSNode, max_depth: int = 4) -> str:
        """
        Generate a visualization of the MCTS search tree.
        
        Args:
            root_node: Root node of the MCTS tree.
            max_depth: Maximum depth to visualize.
            
        Returns:
            Base64-encoded PNG image of the tree visualization.
        """
        try:
            import networkx as nx
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            nodes_to_process = [(root_node, 0, None)]
            node_labels = {}
            node_scores = {}
            node_visits = {}
            
            while nodes_to_process:
                node, depth, parent_id = nodes_to_process.pop(0)
                node_id = node.node_id[:8]  # Short ID for display
                
                # Add the node if not already in the graph
                if node_id not in G:
                    G.add_node(node_id)
                    node_labels[node_id] = f"{node_id}\n{node.avg_reward:.3f}"
                    node_scores[node_id] = node.avg_reward
                    node_visits[node_id] = node.visit_count
                
                # Add edge from parent if it exists
                if parent_id:
                    G.add_edge(parent_id, node_id)
                
                # Process children if not at max depth
                if depth < max_depth:
                    for child in node.children.values():
                        nodes_to_process.append((child, depth + 1, node_id))
            
            # Create the figure
            plt.figure(figsize=(12, 8))
            ax = plt.gca()  # Get current axes for colorbar reference
            
            # Set up the layout
            pos = nx.spring_layout(G)
            
            # Draw nodes with color based on reward
            node_colors = [node_scores[n] for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=[min(2000, max(300, v * 10)) for v in node_visits.values()],
                                  cmap=plt.cm.viridis, vmin=0, vmax=1)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
            
            # Add a title and colorbar
            plt.title(f"MCTS Search Tree (max depth: {max_depth})")
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Reward")  # Fix: Added ax parameter
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode the buffer as base64
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:  # Broader exception handling
            logger.error(f"Error generating tree visualization: {e}")
            return ""
    
    @staticmethod
    def generate_trajectory_visualization(trajectories: List[Dict[str, Any]]) -> str:
        """
        Generate a visualization of the top trajectories.
        
        Args:
            trajectories: List of trajectory data.
            
        Returns:
            Base64-encoded PNG image of the trajectory visualization.
        """
        try:
            # Extract data
            path_scores = [t["evaluation"]["path_score"] for t in trajectories]
            path_lengths = [t["evaluation"]["path_length"] for t in trajectories]
            
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Create bars
            x = np.arange(len(trajectories))
            plt.bar(x, path_scores, width=0.4, label='Path Score')
            
            # Add labels
            plt.xlabel('Trajectory')
            plt.ylabel('Score')
            plt.title('Top Optimization Trajectories')
            plt.xticks(x, [f"Path {i+1}" for i in range(len(trajectories))])
            
            # Add path length as text on each bar
            for i, score in enumerate(path_scores):
                plt.text(i, score + 0.02, f"Length: {path_lengths[i]}", 
                        ha='center', va='bottom')
            
            # Add a grid
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode the buffer as base64
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating trajectory visualization: {e}")
            return ""
    
    @staticmethod
    def generate_reward_progression_visualization(node: MCTSNode) -> str:
        """
        Generate a visualization of reward progression along a path.
        
        Args:
            node: The final node of the path.
            
        Returns:
            Base64-encoded PNG image of the reward progression visualization.
        """
        try:
            # Get the path from root to node
            path = node.get_path_from_root()
            
            # Extract rewards
            rewards = [n.avg_reward for n in path]
            depth = list(range(len(path)))
            
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Create line plot
            plt.plot(depth, rewards, marker='o', linestyle='-', linewidth=2, markersize=8)
            
            # Add labels
            plt.xlabel('Optimization Depth')
            plt.ylabel('Average Reward')
            plt.title('Reward Progression Along Optimization Path')
            
            # Add a grid
            plt.grid(linestyle='--', alpha=0.7)
            
            # Annotate important points
            plt.annotate(f"Start: {rewards[0]:.3f}", xy=(0, rewards[0]),
                       xytext=(10, -20), textcoords='offset points',
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            plt.annotate(f"End: {rewards[-1]:.3f}", xy=(len(rewards)-1, rewards[-1]),
                       xytext=(-10, 20), textcoords='offset points',
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode the buffer as base64
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating reward progression visualization: {e}")
            return ""
    
    @staticmethod
    def generate_component_comparison_visualization(original: Dict[str, bool], optimized: Dict[str, bool]) -> str:
        """
        Generate a visualization comparing original and optimized prompt components.
        
        Args:
            original: Dictionary of component presence in original prompt.
            optimized: Dictionary of component presence in optimized prompt.
            
        Returns:
            Base64-encoded PNG image of the component comparison visualization.
        """
        try:
            # Extract data
            components = list(original.keys())
            original_values = [int(original[c]) for c in components]
            optimized_values = [int(optimized[c]) for c in components]
            
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Set the positions and width for the bars
            x = np.arange(len(components))
            width = 0.35
            
            # Create bars
            plt.bar(x - width/2, original_values, width, label='Original')
            plt.bar(x + width/2, optimized_values, width, label='Optimized')
            
            # Add labels
            plt.xlabel('Component')
            plt.ylabel('Present')
            plt.title('Prompt Component Comparison')
            plt.xticks(x, components)
            plt.yticks([0, 1], ['No', 'Yes'])
            
            # Add a legend
            plt.legend()
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode the buffer as base64
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating component comparison visualization: {e}")
            return ""