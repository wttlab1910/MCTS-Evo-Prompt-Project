"""
Monte Carlo Tree Search (MCTS) engine for prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import random
import re
from app.core.mcts.node import MCTSNode
from app.core.mcts.selection import UCTSelector
from app.core.mcts.expansion import ActionExpander
from app.core.mcts.simulation import PromptSimulator
from app.core.mcts.backprop import Backpropagator
from app.core.mdp.state import PromptState
from app.core.mdp.action import Action, create_action
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction
from app.core.evolution.mutation import PromptMutator
from app.core.evolution.crossover import PromptCrossover
from app.utils.logger import get_logger

logger = get_logger("mcts.engine")

class MCTSEngine:
    """
    Monte Carlo Tree Search engine for prompt optimization.
    
    This class coordinates the selection, expansion, simulation, and backpropagation
    phases of the MCTS algorithm, as well as evolutionary operations.
    """
    
    def __init__(
        self,
        transition: StateTransition,
        reward_function: RewardFunction,
        max_iterations: int = 100,
        time_limit: float = 60.0,
        exploration_weight: float = 1.41,
        max_depth: int = 4,
        max_children_per_expansion: int = 5,
        evolution_config: Optional[Dict[str, Any]] = None,
        action_generator: Optional[Callable[[PromptState], List[Action]]] = None,
        error_feedback_fn: Optional[Callable[[PromptState], List[Action]]] = None
    ):
        """
        Initialize an MCTS engine.
        
        Args:
            transition: State transition function.
            reward_function: Reward function for evaluating states.
            max_iterations: Maximum number of iterations.
            time_limit: Maximum time to run in seconds.
            exploration_weight: Exploration weight for UCT.
            max_depth: Maximum depth of the search tree.
            max_children_per_expansion: Maximum number of children to create per expansion.
            evolution_config: Configuration for evolutionary operations.
            action_generator: Function to generate actions for a state.
            error_feedback_fn: Function to generate error feedback actions.
        """
        self.transition = transition
        self.reward_function = reward_function
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.max_depth = max_depth
        
        # Initialize MCTS components
        self.selector = UCTSelector(exploration_weight=exploration_weight)
        self.expander = ActionExpander(
            transition=transition,
            max_children_per_expansion=max_children_per_expansion
        )
        self.simulator = PromptSimulator(reward_function=reward_function)
        self.backpropagator = Backpropagator()
        
        # Initialize evolutionary components
        self.evolution_config = evolution_config or {
            "mutation_rate": 0.2,
            "crossover_rate": 0.2,
            "error_feedback_rate": 0.6,
            "adaptive_adjustment": True
        }
        
        self.mutator = PromptMutator()
        self.crossover = PromptCrossover()
        
        # Action generator
        self.action_generator = action_generator
        
        # Error feedback function
        self.error_feedback_fn = error_feedback_fn
        
        # Statistics
        self.stats = {
            "total_iterations": 0,
            "total_time": 0.0,
            "evolutionary_operations": 0,
            "mutations": 0,
            "crossovers": 0,
            "error_feedback_actions": 0,
            "knowledge_integrations": 0
        }
        
        # Root node reference for visualization
        self._root_node = None
        
        logger.debug(f"Initialized MCTSEngine with max_iterations={max_iterations}, "
                     f"time_limit={time_limit}, exploration_weight={exploration_weight}, "
                     f"max_depth={max_depth}")
    
    def optimize(self, initial_state: PromptState) -> Tuple[PromptState, Dict[str, Any]]:
        """
        Run the MCTS algorithm to optimize a prompt.
        
        Args:
            initial_state: Initial prompt state.
            
        Returns:
            Tuple of (best_state, statistics).
        """
        # Create root node
        root = MCTSNode(state=initial_state)
        self._root_node = root  # Store for visualization
        
        # Generate initial available actions if action generator is provided
        if self.action_generator:
            available_actions = self.action_generator(initial_state)
        else:
            available_actions = self.expander.generate_actions_for_node(root)
        
        root.set_available_actions(available_actions)
        
        # Initialize statistics
        iterations = 0
        start_time = time.time()
        
        # Run iterations
        while (iterations < self.max_iterations and 
               time.time() - start_time < self.time_limit):
            # Increment iterations
            iterations += 1
            
            # Run one iteration of MCTS
            self._run_iteration(root)
            
            # Adjust operation rates if needed
            if self.evolution_config.get("adaptive_adjustment", True):
                self._adjust_operation_rates(iterations)
            
            # Log progress
            if iterations % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Completed {iterations} iterations in {elapsed:.2f}s. "
                           f"Current best reward: {self._get_best_node(root).avg_reward:.4f}")
        
        # Update statistics
        elapsed = time.time() - start_time
        self.stats["total_iterations"] += iterations
        self.stats["total_time"] += elapsed
        
        # Find best node
        best_node = self._get_best_node(root)
        
        # Return best state and statistics
        stats = {
            "iterations": iterations,
            "time": elapsed,
            "best_reward": best_node.avg_reward,
            "tree_size": self._count_nodes(root),
            "max_depth": self._max_depth(root),
            **self.stats
        }
        
        logger.info(f"Optimization completed: iterations={iterations}, time={elapsed:.2f}s, "
                   f"best_reward={best_node.avg_reward:.4f}")
        
        return best_node.state, stats
    
    def _run_iteration(self, root: MCTSNode) -> None:
        """
        Run a single iteration of the MCTS algorithm.
        
        Args:
            root: Root node of the MCTS tree.
        """
        # Determine operation type based on rates
        operation = self._select_operation_type()
        
        # Add knowledge integration operation if appropriate
        knowledge_integration_strategy = self.evolution_config.get("knowledge_integration_strategy", "adaptive")
        knowledge_items = self.evolution_config.get("domain_knowledge", [])
        
        # Chance to apply knowledge integration
        knowledge_integration_rate = self.evolution_config.get("knowledge_integration_rate", 0.3)
        apply_knowledge = random.random() < knowledge_integration_rate
        
        if apply_knowledge and knowledge_items:
            if knowledge_integration_strategy == "early" and root.visit_count < 10:
                # Apply knowledge early in the search
                self._run_knowledge_integration_iteration(root, knowledge_items)
                return
            elif knowledge_integration_strategy == "error_guided" and operation == "error_feedback":
                # Replace some error feedback operations with knowledge integration
                if random.random() < 0.5:  # 50% chance to replace
                    self._run_knowledge_integration_iteration(root, knowledge_items)
                    return
            elif knowledge_integration_strategy == "interleaved":
                # Randomly interleave knowledge integration
                if random.random() < 0.3:  # 30% chance
                    self._run_knowledge_integration_iteration(root, knowledge_items)
                    return
            elif knowledge_integration_strategy == "adaptive":
                # Apply knowledge based on search progress
                progress = self.stats["total_iterations"] / self.max_iterations
                
                if progress < 0.3:
                    # Early phase: higher chance for knowledge
                    if random.random() < 0.5:
                        self._run_knowledge_integration_iteration(root, knowledge_items)
                        return
                elif progress < 0.7:
                    # Middle phase: moderate chance
                    if random.random() < 0.3:
                        self._run_knowledge_integration_iteration(root, knowledge_items)
                        return
        
        # Run the operation as before
        if operation == "error_feedback":
            # Run error feedback iteration
            self._run_error_feedback_iteration(root)
            self.stats["error_feedback_actions"] += 1
        elif operation == "mutation":
            # Run mutation iteration
            self._run_mutation_iteration(root)
            self.stats["mutations"] += 1
            self.stats["evolutionary_operations"] += 1
        elif operation == "crossover":
            # Run crossover iteration
            self._run_crossover_iteration(root)
            self.stats["crossovers"] += 1
            self.stats["evolutionary_operations"] += 1
        else:
            # Run standard MCTS iteration
            self._run_standard_iteration(root)
    
    def _integrate_domain_knowledge(self, node: MCTSNode, knowledge_items: List[Dict[str, Any]]) -> Optional[MCTSNode]:
        """
        Integrate domain knowledge into a node.
        
        Args:
            node: Node to enhance with knowledge.
            knowledge_items: Knowledge items to integrate.
            
        Returns:
            New node with integrated knowledge, or None if integration failed.
        """
        if not knowledge_items:
            return None
            
        # Select most relevant knowledge item based on current errors
        selected_item = self._select_relevant_knowledge(node, knowledge_items)
        if not selected_item:
            return None
            
        # Create appropriate action based on knowledge type
        knowledge_type = selected_item.get("type", "")
        
        if knowledge_type == "conceptual_knowledge":
            action = create_action("add_domain_knowledge", parameters={
                "knowledge_text": selected_item.get("statement", ""),
                "domain": selected_item.get("metadata", {}).get("domain", "general")
            })
        elif knowledge_type == "procedural_knowledge":
            steps = selected_item.get("procedure_steps", [])
            if steps:
                action = create_action("modify_workflow", parameters={
                    "steps": steps
                })
            else:
                action = create_action("add_explanation", parameters={
                    "explanation_text": selected_item.get("statement", ""),
                    "target": "task"
                })
        elif knowledge_type == "entity_classification":
            action = create_action("clarify_terminology", parameters={
                "term": selected_item.get("entities", [""])[0] if selected_item.get("entities") else "",
                "definition": selected_item.get("statement", "")
            })
        elif knowledge_type == "format_specification":
            action = create_action("specify_format", parameters={
                "format_text": selected_item.get("statement", "")
            })
        elif knowledge_type == "task_instruction":
            # 修改此处：使用add_rule而不是add_instruction
            action = create_action("add_rule", parameters={
                "rule_text": selected_item.get("statement", ""),
                "priority": "high"
            })
        else:
            # Default to adding as domain knowledge
            action = create_action("add_domain_knowledge", parameters={
                "knowledge_text": selected_item.get("statement", ""),
                "domain": "general"
            })
        
        # Apply action to create new node
        if action.is_applicable(node.state):
            child_node = self.expander.expand_with_action(node, action)
            
            if child_node:
                # Mark as knowledge integration
                child_node.add_evolution_operation(f"knowledge_integration_{knowledge_type}")
                
                # Simulate and backpropagate
                reward = self.simulator.simulate(child_node)
                self.backpropagator.backpropagate(child_node, reward)
                
                return child_node
        
        return None

    def _select_relevant_knowledge(self, node: MCTSNode, knowledge_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the most relevant knowledge item for the current node state.
        
        Args:
            node: Current node.
            knowledge_items: Available knowledge items.
            
        Returns:
            Most relevant knowledge item or None if none is relevant.
        """
        # Get node state and errors
        state = node.state
        errors = node.error_feedback
        
        # If no errors, select based on text content
        if not errors:
            # Extract key terms from state text
            text = state.text.lower()
            terms = set(re.findall(r'\b[a-z]{4,}\b', text))
            
            # Score knowledge items by term overlap
            scores = []
            for item in knowledge_items:
                item_text = item.get("statement", "").lower()
                item_terms = set(re.findall(r'\b[a-z]{4,}\b', item_text))
                
                overlap = len(terms.intersection(item_terms))
                scores.append((item, overlap))
            
            # Return highest scoring item
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0] if scores else None
        
        # If errors exist, select based on error patterns
        else:
            # Extract error types and entities
            error_types = set()
            error_entities = set()
            
            for error in errors:
                error_types.add(error.get("type", ""))
                for entity in error.get("entities", []):
                    error_entities.add(entity)
            
            # Score knowledge items by relevance to errors
            scores = []
            for item in knowledge_items:
                score = 0
                
                # Check for type matches
                if item.get("type", "") in error_types:
                    score += 3
                    
                # Check for entity matches
                item_entities = set(item.get("entities", []))
                entity_overlap = len(error_entities.intersection(item_entities))
                score += entity_overlap * 2
                
                scores.append((item, score))
            
            # Return highest scoring item
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0] if scores and scores[0][1] > 0 else None

    def _run_knowledge_integration_iteration(self, root: MCTSNode, knowledge_items: List[Dict[str, Any]]) -> None:
        """
        Run a knowledge integration iteration.
        
        Args:
            root: Root node of the MCTS tree.
            knowledge_items: Domain knowledge items to potentially integrate.
        """
        # Selection: select a promising node to enhance with knowledge
        selected_node = self._select(root)
        
        # Try to integrate knowledge
        enhanced_node = self._integrate_domain_knowledge(selected_node, knowledge_items)
        
        # If integration was successful, increment counter
        if enhanced_node:
            self.stats["knowledge_integrations"] += 1
        # If integration failed, fall back to standard iteration
        else:
            self._run_standard_iteration(root)
    
    def _run_standard_iteration(self, root: MCTSNode) -> None:
        """
        Run a standard MCTS iteration (selection, expansion, simulation, backpropagation).
        
        Args:
            root: Root node of the MCTS tree.
        """
        # Selection: select a promising node to expand
        selected_node = self._select(root)
        
        # Expansion: create new child nodes by applying actions
        if self.action_generator:
            available_actions = self.action_generator(selected_node.state)
        else:
            available_actions = self.expander.generate_actions_for_node(selected_node)
        
        expanded_nodes = self._expand(selected_node, available_actions)
        
        # If no nodes were expanded, use the selected node
        if not expanded_nodes:
            nodes_to_simulate = [selected_node]
        else:
            nodes_to_simulate = expanded_nodes
        
        # Simulation and Backpropagation: evaluate each expanded node and update statistics
        for node in nodes_to_simulate:
            reward = self.simulator.simulate(node)
            self.backpropagator.backpropagate(node, reward)
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Select a promising node to expand, respecting the max_depth constraint.
        
        Args:
            root: Root node of the MCTS tree.
            
        Returns:
            Selected node.
        """
        # Use the selector to find a promising node
        current = self.selector.select(root)
        
        # If the current node is too deep, backtrack to a shallower node
        while current.generation > self.max_depth:
            current = current.parent or root
        
        return current
    
    def _expand(self, node: MCTSNode, available_actions: List[Action]) -> List[MCTSNode]:
        """
        Expand a node by applying actions, respecting the max_depth constraint.
        
        Args:
            node: Node to expand.
            available_actions: List of actions that can be applied.
            
        Returns:
            List of newly created child nodes.
        """
        # Don't expand beyond max_depth
        if node.generation >= self.max_depth:
            return []
            
        # Use the expander to create new child nodes
        return self.expander.expand(node, available_actions)
    
    def _run_error_feedback_iteration(self, root: MCTSNode) -> None:
        """
        Run an error feedback iteration, which focuses on addressing specific errors.
        
        Args:
            root: Root node of the MCTS tree.
        """
        # Select a promising node to improve, respecting max_depth
        selected_node = self._select(root)
        
        # Don't apply error feedback beyond max_depth
        if selected_node.generation >= self.max_depth:
            self._run_standard_iteration(root)
            return
        
        # Check if we have a custom error feedback function
        if self.error_feedback_fn:
            # Use the provided error feedback function to generate actions
            try:
                feedback_actions = self.error_feedback_fn(selected_node.state)
                
                if feedback_actions:
                    # Apply one of the feedback actions
                    action = random.choice(feedback_actions)
                    if action.is_applicable(selected_node.state):
                        child_node = self.expander.expand_with_action(selected_node, action)
                        
                        if child_node:
                            # Mark as error feedback
                            child_node.add_error_feedback({
                                "type": "error_feedback",
                                "description": "Error-based feedback from evaluation",
                                "action": str(action)
                            })
                            
                            # Simulate and backpropagate
                            reward = self.simulator.simulate(child_node)
                            self.backpropagator.backpropagate(child_node, reward)
                            return
            except Exception as e:
                logger.error(f"Error during error feedback: {e}")
        
        # Fallback to synthetic error feedback if no custom function or it failed
        error_actions = [
            create_action("add_constraint", parameters={
                "constraint_text": "Ensure all information is factually accurate",
                "location": "separate_section"
            }),
            create_action("add_explanation", parameters={
                "explanation_text": "Pay special attention to edge cases",
                "target": "step",
                "target_index": 0
            }),
            create_action("add_rule", parameters={
                "rule_text": "Verify data consistency before proceeding",
                "priority": "high"
            })
        ]
        
        # Apply one of the error feedback actions
        action = random.choice(error_actions)
        if action.is_applicable(selected_node.state):
            child_node = self.expander.expand_with_action(selected_node, action)
            
            if child_node:
                # Mark as error feedback
                child_node.add_error_feedback({
                    "type": "synthetic_error",
                    "description": "Synthetic error for demonstration",
                    "action": str(action)
                })
                
                # Simulate and backpropagate
                reward = self.simulator.simulate(child_node)
                self.backpropagator.backpropagate(child_node, reward)
        else:
            # If action is not applicable, run a standard iteration
            self._run_standard_iteration(root)
    
    def _run_mutation_iteration(self, root: MCTSNode) -> None:
        """
        Run a mutation iteration, which applies random variations to promising nodes.
        
        Args:
            root: Root node of the MCTS tree.
        """
        # Select a node to mutate (favor nodes with higher rewards), respecting max_depth
        selected_node = self._select_node_for_evolution(root)
        
        # Don't apply mutation beyond max_depth
        if selected_node.generation >= self.max_depth:
            self._run_standard_iteration(root)
            return
        
        # Apply mutation
        mutated_state = self.mutator.mutate(selected_node.state)
        
        # Create a new action representing the mutation
        mutation_action = create_action("mutation", parameters={
            "mutation_type": "random_variation",
            "description": "Applied random mutation to prompt"
        })
        
        # Add the mutated state as a child
        child_node = selected_node.add_child(mutation_action, mutated_state)
        child_node.add_evolution_operation("mutation")
        
        # Simulate and backpropagate
        reward = self.simulator.simulate(child_node)
        self.backpropagator.backpropagate(child_node, reward)
    
    def _run_crossover_iteration(self, root: MCTSNode) -> None:
        """
        Run a crossover iteration, which combines elements from two promising nodes.
        
        Args:
            root: Root node of the MCTS tree.
        """
        # Need at least 2 nodes with visits for crossover
        if root.visit_count < 2:
            self._run_standard_iteration(root)
            return
        
        # Select two parent nodes for crossover, respecting max_depth
        parent1 = self._select_node_for_evolution(root)
        
        # Don't apply crossover beyond max_depth
        if parent1.generation >= self.max_depth:
            self._run_standard_iteration(root)
            return
        
        parent2 = self._select_node_for_evolution(root, exclude=parent1)
        
        # Apply crossover
        crossover_state = self.crossover.crossover(parent1.state, parent2.state)
        
        # Create a new action representing the crossover
        crossover_action = create_action("crossover", parameters={
            "crossover_type": "component_recombination",
            "description": "Combined elements from two parent prompts"
        })
        
        # Add the crossover state as a child of parent1
        child_node = parent1.add_child(crossover_action, crossover_state)
        child_node.add_evolution_operation(f"crossover with {parent2.node_id[:8]}")
        
        # Simulate and backpropagate
        reward = self.simulator.simulate(child_node)
        self.backpropagator.backpropagate(child_node, reward)
    
    def _select_operation_type(self) -> str:
        """
        Select an operation type based on configured rates.
        
        Returns:
            Operation type: "error_feedback", "mutation", "crossover", or "standard".
        """
        # Get current rates
        mutation_rate = self.evolution_config.get("mutation_rate", 0.2)
        crossover_rate = self.evolution_config.get("crossover_rate", 0.2)
        error_feedback_rate = self.evolution_config.get("error_feedback_rate", 0.6)
        
        # Normalize rates to ensure they sum to 1.0
        total = mutation_rate + crossover_rate + error_feedback_rate
        if total > 0:
            mutation_rate /= total
            crossover_rate /= total
            error_feedback_rate /= total
        
        # Select operation based on rates
        r = random.random()
        if r < error_feedback_rate:
            return "error_feedback"
        elif r < error_feedback_rate + mutation_rate:
            return "mutation"
        elif r < error_feedback_rate + mutation_rate + crossover_rate:
            return "crossover"
        else:
            return "standard"
    
    def _adjust_operation_rates(self, iteration: int) -> None:
        """
        Adjust operation rates based on the current iteration and performance.
        
        Args:
            iteration: Current iteration number.
        """
        # Calculate progress as a percentage of max iterations
        progress = iteration / self.max_iterations
        
        # Early stage (first 30% of iterations)
        if progress < 0.3:
            mutation_rate = 0.3
            crossover_rate = 0.3
            error_feedback_rate = 0.4
        # Middle stage (30%-70% of iterations)
        elif progress < 0.7:
            mutation_rate = 0.2
            crossover_rate = 0.2
            error_feedback_rate = 0.6
        # Late stage (final 30% of iterations)
        else:
            mutation_rate = 0.1
            crossover_rate = 0.1
            error_feedback_rate = 0.8
        
        # Update rates
        self.evolution_config["mutation_rate"] = mutation_rate
        self.evolution_config["crossover_rate"] = crossover_rate
        self.evolution_config["error_feedback_rate"] = error_feedback_rate
        
        logger.debug(f"Adjusted operation rates: mutation={mutation_rate:.2f}, "
                    f"crossover={crossover_rate:.2f}, error_feedback={error_feedback_rate:.2f}")
    
    def _select_node_for_evolution(
        self, 
        root: MCTSNode, 
        exclude: Optional[MCTSNode] = None
    ) -> MCTSNode:
        """
        Select a node for evolution operations, favoring nodes with higher rewards.
        
        Args:
            root: Root node of the MCTS tree.
            exclude: Node to exclude from selection (for second parent in crossover).
            
        Returns:
            Selected node.
        """
        # Collect all nodes with at least one visit
        nodes = []
        self._collect_nodes_with_visits(root, nodes, exclude=exclude)
        
        # Filter nodes to respect max_depth constraint for adding children
        valid_nodes = [node for node in nodes if node.generation < self.max_depth]
        
        if not valid_nodes:
            return root
        
        # Weight selection by reward
        weights = [max(0.01, node.avg_reward) for node in valid_nodes]
        total_weight = sum(weights)
        
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            return random.choices(valid_nodes, weights=weights, k=1)[0]
        else:
            return random.choice(valid_nodes)
    
    def _collect_nodes_with_visits(
        self, 
        node: MCTSNode, 
        result: List[MCTSNode],
        exclude: Optional[MCTSNode] = None
    ) -> None:
        """
        Collect all nodes in the tree that have been visited at least once.
        
        Args:
            node: Current node.
            result: List to collect nodes in.
            exclude: Node to exclude from collection.
        """
        if node.visit_count > 0 and node != exclude:
            result.append(node)
        
        for child in node.children.values():
            self._collect_nodes_with_visits(child, result, exclude=exclude)
    
    def _get_best_node(self, root: MCTSNode) -> MCTSNode:
        """
        Find the node with the highest average reward in the tree.
        
        Args:
            root: Root node of the MCTS tree.
            
        Returns:
            Node with the highest average reward.
        """
        best_node = root
        best_reward = root.avg_reward
        
        # Collect all nodes with at least one visit
        nodes = []
        self._collect_nodes_with_visits(root, nodes)
        
        # Find node with highest average reward
        for node in nodes:
            if node.visit_count > 0 and node.avg_reward > best_reward:
                best_node = node
                best_reward = node.avg_reward
        
        return best_node
    
    def _count_nodes(self, node: MCTSNode) -> int:
        """
        Count the number of nodes in the tree.
        
        Args:
            node: Root node of the tree.
            
        Returns:
            Number of nodes.
        """
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
    
    def _max_depth(self, node: MCTSNode) -> int:
        """
        Calculate the maximum depth of the tree.
        
        Args:
            node: Root node of the tree.
            
        Returns:
            Maximum depth.
        """
        if not node.children:
            return 0
        
        return 1 + max(self._max_depth(child) for child in node.children.values())
    
    def _get_root_node(self) -> Optional[MCTSNode]:
        """
        Get the root node of the current MCTS tree.
        For visualization and debugging purposes.
        
        Returns:
            Root node or None if no tree has been built.
        """
        return self._root_node