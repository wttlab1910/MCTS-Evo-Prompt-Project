"""
Phase3
Demonstration of MCTS Strategic Planning with Evolutionary Algorithms.
"""
import sys
import os
import time
import random
from colorama import init, Fore, Style

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Ensure backend directory is also in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction, PerformanceEvaluator

from app.core.mcts.node import MCTSNode
from app.core.mcts.selection import UCTSelector
from app.core.mcts.expansion import ActionExpander
from app.core.mcts.simulation import PromptSimulator
from app.core.mcts.backprop import Backpropagator
from app.core.mcts.engine import MCTSEngine

from app.core.evolution.mutation import PromptMutator
from app.core.evolution.crossover import PromptCrossover
from app.core.evolution.selection import EvolutionSelector

from app.knowledge.error.error_collector import ErrorCollector
from app.knowledge.error.error_analyzer import ErrorAnalyzer
from app.knowledge.error.feedback_generator import FeedbackGenerator

# Initialize colorama
init()

def print_header(text):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 80}")
    print(f"{text.center(80)}")
    print(f"{'=' * 80}{Style.RESET_ALL}")

def print_section(text):
    """Print colored section title"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_state(state):
    """Print a prompt state with highlighting"""
    print(f"{Fore.GREEN}Prompt State ID: {state.state_id[:8]}{Style.RESET_ALL}")
    
    # Print text with some formatting
    lines = state.text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("Role:"):
            print(f"{Fore.MAGENTA}{line}{Style.RESET_ALL}")
        elif line.startswith("Task:"):
            print(f"{Fore.BLUE}{line}{Style.RESET_ALL}")
        elif line.startswith("-"):
            print(f"{Fore.CYAN}  {line}{Style.RESET_ALL}")
        elif line.startswith("Output Format:"):
            print(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
        else:
            print(f"{Fore.WHITE}{line}{Style.RESET_ALL}")
    
    # Print metrics
    reward_fn = RewardFunction(
        task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
    )
    reward = reward_fn.calculate(state)
    print(f"\n{Fore.CYAN}Reward: {reward:.4f}{Style.RESET_ALL}")

def print_node(node, depth=0):
    """Print an MCTS node with highlighting"""
    indent = "  " * depth
    
    # Color based on reward (green for high, yellow for medium, red for low)
    if node.visit_count > 0:
        if node.avg_reward > 0.7:
            color = Fore.GREEN
        elif node.avg_reward > 0.4:
            color = Fore.YELLOW
        else:
            color = Fore.RED
    else:
        color = Fore.WHITE
    
    # Print node information
    print(f"{indent}{color}Node {node.node_id[:8]}: visits={node.visit_count}, "
          f"avg_reward={node.avg_reward:.4f}, children={len(node.children)}{Style.RESET_ALL}")
    
    if node.action_applied:
        print(f"{indent}  Action: {Fore.CYAN}{node.action_applied}{Style.RESET_ALL}")
    
    # Print evolution operations if any
    if node.evolution_history:
        print(f"{indent}  Evolution: {Fore.MAGENTA}{node.evolution_history[-1]}{Style.RESET_ALL}")

def print_mcts_tree(root, max_depth=3, current_depth=0):
    """打印MCTS树结构，显示节点和子节点关系"""
    if current_depth > max_depth:
        print(f"{'  ' * current_depth}... (树太深，已截断)")
        return
        
    print_node(root, current_depth)
    
    # 显示子节点统计
    if root.children:
        num_children = len(root.children)
        node_rewards = [c.avg_reward for c in root.children.values() if c.visit_count > 0]
        if node_rewards:
            max_reward = max(node_rewards)
            min_reward = min(node_rewards)
            
            print(f"{'  ' * current_depth}└─ {num_children} 子节点, 奖励范围: [{min_reward:.4f} - {max_reward:.4f}]")
            
            # 只显示前3个子节点，按奖励排序
            sorted_children = sorted(root.children.values(), key=lambda x: x.avg_reward if x.visit_count > 0 else 0, reverse=True)
            for i, child in enumerate(sorted_children[:3]):
                is_last = (i == len(sorted_children[:3]) - 1)
                prefix = "└─" if is_last else "├─"
                print(f"{'  ' * (current_depth+1)}{prefix} 子节点 {i+1}/{min(3, len(sorted_children))}:")
                print_mcts_tree(child, max_depth, current_depth + 2)
            
            if len(sorted_children) > 3:
                print(f"{'  ' * (current_depth+1)}└─ ... 还有 {len(sorted_children)-3} 个子节点")
        else:
            print(f"{'  ' * current_depth}└─ {num_children} 子节点 (均未访问)")

def print_reward_histogram(rewards, bins=10, width=40):
    """打印ASCII柱状图显示奖励分布"""
    if not rewards:
        print("\n奖励分布: 没有足够的数据")
        return
        
    print("\n奖励分布:")
    
    # 计算直方图数据
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    if min_reward == max_reward:
        print(f"所有奖励值都是: {min_reward:.4f}")
        return
        
    bin_width = (max_reward - min_reward) / bins
    
    # 统计每个bin的数量
    counts = [0] * bins
    for reward in rewards:
        bin_idx = min(bins - 1, int((reward - min_reward) / bin_width))
        counts[bin_idx] += 1
    
    # 找到最大计数来缩放
    max_count = max(counts) if counts else 0
    scale = width / max_count if max_count > 0 else 1
    
    # 打印柱状图
    for i in range(bins):
        bin_min = min_reward + i * bin_width
        bin_max = bin_min + bin_width
        bar_length = int(counts[i] * scale)
        bar = "#" * bar_length
        print(f"{bin_min:.2f}-{bin_max:.2f} [{counts[i]:3d}]: {bar}")

def print_operation_stats(stats):
    """打印操作统计表格"""
    print("\n操作统计:")
    
    # 计算百分比
    total_ops = sum([
        stats.get("mutations", 0),
        stats.get("crossovers", 0),
        stats.get("error_feedback_actions", 0)
    ])
    
    if total_ops == 0:
        print("没有执行任何操作")
        return
    
    # 创建表格行
    headers = ["操作类型", "数量", "百分比", "图表"]
    rows = [
        ["突变 (Mutation)", stats.get("mutations", 0), 
         f"{stats.get('mutations', 0)/total_ops*100:.1f}%" if total_ops else "0%",
         "=" * int(stats.get("mutations", 0)/total_ops*20) if total_ops else ""],
        
        ["交叉 (Crossover)", stats.get("crossovers", 0),
         f"{stats.get('crossovers', 0)/total_ops*100:.1f}%" if total_ops else "0%",
         "=" * int(stats.get("crossovers", 0)/total_ops*20) if total_ops else ""],
        
        ["错误反馈", stats.get("error_feedback_actions", 0),
         f"{stats.get('error_feedback_actions', 0)/total_ops*100:.1f}%" if total_ops else "0%",
         "=" * int(stats.get("error_feedback_actions", 0)/total_ops*20) if total_ops else ""]
    ]
    
    # 打印表格
    col_widths = [20, 8, 10, 22]
    print("-" * sum(col_widths))
    print("".join(word.ljust(col_widths[i]) for i, word in enumerate(headers)))
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(item).ljust(col_widths[i]) for i, item in enumerate(row)))
    print("-" * sum(col_widths))

def simulate_typing(text, delay=0.01):
    """Simulate typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo_mcts_node():
    """Demonstrate MCTS node functionality."""
    print_header("MCTS Node Demonstration")
    
    # Create a basic state
    basic_text = """
    Task: Analyze the sentiment of the text.
    """
    
    state = PromptState(basic_text)
    
    print_section("Creating Root Node")
    root = MCTSNode(state=state)
    print_node(root)
    
    print_section("Adding Children")
    transition = StateTransition()
    
    # Add a child with AddRoleAction
    action1 = create_action("add_role", parameters={"role_text": "Sentiment Analysis Expert"})
    child_state1 = transition.apply(state, action1)
    child1 = root.add_child(action1, child_state1)
    
    # Add a child with ModifyWorkflowAction
    action2 = create_action("modify_workflow", parameters={
        "steps": [
            "Read the text carefully",
            "Identify sentiment-bearing words",
            "Determine overall sentiment"
        ]
    })
    child_state2 = transition.apply(state, action2)
    child2 = root.add_child(action2, child_state2)
    
    # Display tree
    print_node(root)
    print_node(child1, 1)
    print_node(child2, 1)
    
    print_section("Updating Statistics")
    # Simulate visits and rewards
    root.update_statistics(0.5)
    child1.update_statistics(0.7)
    child2.update_statistics(0.4)
    
    # Update again with different rewards
    root.update_statistics(0.6)
    child1.update_statistics(0.8)
    child2.update_statistics(0.3)
    
    # Display updated tree
    print_node(root)
    print_node(child1, 1)
    print_node(child2, 1)
    
    print_section("Paths from Root")
    path = child1.get_path_from_root()
    print(f"Path for node {child1.node_id[:8]}: {' -> '.join([n.node_id[:8] for n in path])}")
    
    action_path = child1.get_action_path_from_root()
    print(f"Actions: {' -> '.join([str(a) for a in action_path if a])}")

def demo_selection_expansion():
    """Demonstrate selection and expansion functionality."""
    print_header("Selection and Expansion Demonstration")
    
    # Create initial state and tree
    basic_text = """
    Task: Analyze the sentiment of the text.
    """
    
    state = PromptState(basic_text)
    transition = StateTransition()
    reward_fn = RewardFunction(
        task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
    )
    
    print_section("Creating Initial Tree")
    root = MCTSNode(state=state)
    
    # Add some children with varying statistics
    actions = [
        create_action("add_role", parameters={"role_text": "Sentiment Analysis Expert"}),
        create_action("modify_workflow", parameters={
            "steps": [
                "Read the text carefully",
                "Identify sentiment-bearing words",
                "Determine overall sentiment"
            ]
        }),
        create_action("specify_format", parameters={
            "format_text": "Provide sentiment as positive, negative, or neutral"
        })
    ]
    
    children = []
    for action in actions:
        child_state = transition.apply(state, action)
        child = root.add_child(action, child_state)
        children.append(child)
    
    # Set up visits and rewards
    visits = [3, 10, 1]
    rewards = [0.7, 0.5, 0.3]
    
    root.update_statistics(0.0)  # Root needs at least one visit
    
    for child, visit_count, reward in zip(children, visits, rewards):
        for _ in range(visit_count):
            child.update_statistics(reward)
    
    # Display tree
    print_node(root)
    for child in children:
        print_node(child, 1)
    
    print_section("UCT Selection")
    selector = UCTSelector(exploration_weight=1.41)
    
    # Select with different exploration weights
    explorative = UCTSelector(exploration_weight=2.0)
    exploitative = UCTSelector(exploration_weight=0.1)
    
    selected_balanced = selector.select(root)
    selected_explorative = explorative.select(root)
    selected_exploitative = exploitative.select(root)
    
    print(f"Balanced selection (w=1.41): Node {selected_balanced.node_id[:8]}")
    print(f"Explorative selection (w=2.0): Node {selected_explorative.node_id[:8]}")
    print(f"Exploitative selection (w=0.1): Node {selected_exploitative.node_id[:8]}")
    
    print_section("Action Expansion")
    expander = ActionExpander(transition=transition, max_children_per_expansion=2)
    
    # Generate some actions
    more_actions = [
        create_action("add_example", parameters={
            "example_text": "Input: 'I love this product!' Output: Positive",
            "example_type": "input_output"
        }),
        create_action("add_constraint", parameters={
            "constraint_text": "Only use information from the text"
        }),
        create_action("add_explanation", parameters={
            "explanation_text": "Focus on emotion-bearing words",
            "target": "task"
        })
    ]
    
    # Expand with some actions
    new_children = expander.expand(selected_balanced, more_actions)
    
    print(f"Expanded node {selected_balanced.node_id[:8]} with {len(new_children)} new children")
    for child in new_children:
        print_node(child, 1)

def demo_simulation_backpropagation():
    """Demonstrate simulation and backpropagation functionality."""
    print_header("Simulation and Backpropagation Demonstration")
    
    # Create initial state and tree
    basic_text = """
    Task: Analyze the sentiment of the text.
    """
    
    state = PromptState(basic_text)
    transition = StateTransition()
    reward_fn = RewardFunction(
        task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
    )
    
    print_section("Creating Test Tree")
    root = MCTSNode(state=state)
    
    # Create a few actions to apply
    actions = [
        create_action("add_role", parameters={"role_text": "Sentiment Analysis Expert"}),
        create_action("modify_workflow", parameters={
            "steps": [
                "Read the text carefully",
                "Identify sentiment-bearing words",
                "Determine overall sentiment"
            ]
        }),
        create_action("specify_format", parameters={
            "format_text": "Provide sentiment as positive, negative, or neutral"
        })
    ]
    
    # Create a simple path through the tree
    current = root
    path = [current]
    
    for action in actions:
        child_state = transition.apply(current.state, action)
        child = current.add_child(action, child_state)
        path.append(child)
        current = child
    
    # Display tree
    for i, node in enumerate(path):
        print_node(node, i)
    
    print_section("Simulation")
    simulator = PromptSimulator(reward_function=reward_fn)
    
    # Simulate leaf node
    leaf = path[-1]
    reward = simulator.simulate(leaf)
    
    print(f"Simulated node {leaf.node_id[:8]} with reward: {reward:.4f}")
    
    print_section("Backpropagation")
    backpropagator = Backpropagator()
    
    # Display tree before backpropagation
    print("Before backpropagation:")
    for i, node in enumerate(path):
        print_node(node, i)
    
    # Backpropagate the reward
    backpropagator.backpropagate(leaf, reward)
    
    # Display tree after backpropagation
    print("\nAfter backpropagation:")
    for i, node in enumerate(path):
        print_node(node, i)
    
    # Simulate and backpropagate for other nodes
    print("\nAdditional simulations and backpropagations:")
    for i, node in enumerate(path[1:-1]):  # Skip root and leaf
        reward = simulator.simulate(node)
        print(f"Simulated node {node.node_id[:8]} with reward: {reward:.4f}")
        backpropagator.backpropagate(node, reward)
    
    # Display final tree
    print("\nFinal tree:")
    for i, node in enumerate(path):
        print_node(node, i)

def demo_mcts_engine():
    """Demonstrate the complete MCTS engine."""
    print_header("MCTS Engine Demonstration")
    
    # Create initial state
    basic_text = """
    Task: Analyze the sentiment of the text.
    """
    
    state = PromptState(basic_text)
    transition = StateTransition()
    reward_fn = RewardFunction(
        task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
    )
    
    print_section("Initial State")
    print_state(state)
    
    print_section("Running MCTS Optimization")
    mcts_engine = MCTSEngine(
        transition=transition,
        reward_function=reward_fn,
        max_iterations=30,
        time_limit=5.0,
        exploration_weight=1.41,
        max_children_per_expansion=3
    )
    
    # Run optimization
    print("Optimizing prompt...")
    best_state, stats = mcts_engine.optimize(state)
    
    print_section("Optimization Statistics")
    print(f"Iterations: {stats['iterations']}")
    print(f"Time: {stats['time']:.2f} seconds")
    print(f"Best reward: {stats['best_reward']:.4f}")
    print(f"Tree size: {stats['tree_size']} nodes")
    print(f"Max depth: {stats['max_depth']}")
    print(f"Total evolutionary operations: {stats['evolutionary_operations']}")
    print(f"Mutations: {stats['mutations']}")
    print(f"Crossovers: {stats['crossovers']}")
    print(f"Error feedback actions: {stats['error_feedback_actions']}")
    
    # 添加可视化部分
    print_section("MCTS 树结构")
    root = mcts_engine._get_root_node()
    if root:
        print_mcts_tree(root)
        
        # 收集所有节点的奖励
        all_rewards = []
        def collect_rewards(node):
            if node.visit_count > 0:
                all_rewards.append(node.avg_reward)
            for child in node.children.values():
                collect_rewards(child)
                
        collect_rewards(root)
        print_reward_histogram(all_rewards)
    
    # 打印操作统计
    print_operation_stats(stats)
    
    print_section("Best Prompt Found")
    print_state(best_state)
    
    # Compare with initial state
    initial_reward = reward_fn.calculate(state)
    best_reward = reward_fn.calculate(best_state)
    improvement = best_reward - initial_reward
    
    print(f"\nImprovement: {improvement:.4f} ({initial_reward:.4f} → {best_reward:.4f})")

def demo_evolutionary_operations():
    """Demonstrate evolutionary operations."""
    print_header("Evolutionary Operations Demonstration")
    
    # Create a structured state
    structured_text = """
    Role: Sentiment Analysis Expert
    Task: Analyze the sentiment of the provided text.
    
    Steps:
    - Read the text carefully
    - Identify sentiment-bearing words and phrases
    - Determine overall sentiment
    
    Output Format: Provide sentiment as positive, negative, or neutral.
    """
    
    structured_state = PromptState(structured_text)
    
    # Create a second state to use in crossover
    alternative_text = """
    Role: Text Analyzer
    Task: Determine the emotional tone of the given text.
    
    Steps:
    - Process the text thoroughly
    - Extract emotional indicators
    - Evaluate overall sentiment
    - Provide confidence level
    
    Output Format: Return the sentiment (positive/negative/neutral) with confidence score.
    """
    
    alternative_state = PromptState(alternative_text)
    
    print_section("Original States")
    print("State 1:")
    print_state(structured_state)
    
    print("\nState 2:")
    print_state(alternative_state)
    
    print_section("Mutation")
    mutator = PromptMutator(mutation_strength=0.5)
    
    # Apply different mutations
    mutations = []
    mutation_types = [
        "_vocabulary_substitution",
        "_structural_variation",
        "_content_reduction",
        "_content_expansion"
    ]
    
    for mutation_type in mutation_types:
        mutation_fn = getattr(mutator, mutation_type)
        mutated_text = mutation_fn(structured_state.text)
        mutated_state = PromptState(
            text=mutated_text,
            history=structured_state.history + [mutation_type],
            parent=structured_state,
            action_applied="mutation"
        )
        mutations.append((mutation_type, mutated_state))
    
    # Display mutations
    for mutation_type, mutated_state in mutations:
        print(f"\n{Fore.MAGENTA}Mutation: {mutation_type}{Style.RESET_ALL}")
        print_state(mutated_state)
    
    print_section("Crossover")
    crossover = PromptCrossover()
    
    # Apply different crossovers
    crossovers = []
    crossover_types = [
        "_component_crossover",
        "_section_crossover",
        "_alternating_lines_crossover"
    ]
    
    for crossover_type in crossover_types:
        crossover_fn = getattr(crossover, crossover_type)
        crossed_state = PromptState(
            text=crossover_fn(structured_state, alternative_state),
            history=structured_state.history + [f"crossover with {alternative_state.state_id[:8]}"],
            parent=structured_state,
            action_applied="crossover"
        )
        crossovers.append((crossover_type, crossed_state))
    
    # Display crossovers
    for crossover_type, crossed_state in crossovers:
        print(f"\n{Fore.MAGENTA}Crossover: {crossover_type}{Style.RESET_ALL}")
        print_state(crossed_state)
    
    print_section("Evolution Selection")
    selector = EvolutionSelector()
    
    # Create some nodes with different rewards
    nodes = []
    for i in range(5):
        node = MCTSNode(state=structured_state)
        # Set different rewards
        reward = 0.2 + (i * 0.2)  # 0.2, 0.4, 0.6, 0.8, 1.0
        node.update_statistics(reward)
        nodes.append(node)
    
    # Test different selection methods
    tournament_selected = selector.tournament_select(nodes, tournament_size=3)
    roulette_selected = selector.roulette_wheel_select(nodes)
    rank_selected = selector.rank_select(nodes)
    
    print(f"Tournament selection: Node {tournament_selected.node_id[:8]} with reward {tournament_selected.avg_reward:.4f}")
    print(f"Roulette wheel selection: Node {roulette_selected.node_id[:8]} with reward {roulette_selected.avg_reward:.4f}")
    print(f"Rank selection: Node {rank_selected.node_id[:8]} with reward {rank_selected.avg_reward:.4f}")
    
    # Select a diverse pair
    parent1, parent2 = selector.select_diverse_pair(nodes)
    print(f"Diverse pair: Node {parent1.node_id[:8]} and Node {parent2.node_id[:8]}")

def demo_error_feedback():
    """Demonstrate error feedback system."""
    print_header("Error Feedback System Demonstration")
    
    # Create a basic state
    basic_text = """
    Task: Analyze the sentiment of the text.
    """
    
    state = PromptState(basic_text)
    
    print_section("Initial State")
    print_state(state)
    
    print_section("Error Collection")
    error_collector = ErrorCollector()
    
    # Create mock examples
    examples = [
        {"text": "I love this product!", "expected": "positive"},
        {"text": "This is terrible service.", "expected": "negative"},
        {"text": "The item arrived on time.", "expected": "neutral"},
        {"text": "I'm very disappointed.", "expected": "negative"},
        {"text": "Amazing experience!", "expected": "positive"}
    ]
    
    # Collect errors
    errors = error_collector.collect_errors(state, examples)
    
    # Display errors
    print(f"Collected {len(errors)} errors:")
    for i, error in enumerate(errors):
        print(f"\n{Fore.RED}Error {i+1}:{Style.RESET_ALL}")
        print(f"  Example: {error['example']['text']}")
        print(f"  Expected: {error.get('expected', 'Unknown')}")
        print(f"  Error Type: {error['error_type']}")
    
    print_section("Error Analysis")
    error_analyzer = ErrorAnalyzer()
    
    # Analyze errors
    analysis = error_analyzer.analyze_errors(errors)
    
    # Display analysis
    print(f"{Fore.YELLOW}Summary: {analysis['summary']}{Style.RESET_ALL}")
    
    print("\nError Clusters:")
    for category, cluster_errors in analysis['error_clusters'].items():
        description = error_analyzer.get_error_category_description(category)
        print(f"  {Fore.CYAN}{category}{Style.RESET_ALL} ({len(cluster_errors)} errors): {description}")
    
    print("\nPatterns:")
    for pattern in analysis['patterns']:
        print(f"  {Fore.MAGENTA}{pattern['pattern_type']}{Style.RESET_ALL}: {pattern['description']}")
    
    print_section("Feedback Generation")
    feedback_generator = FeedbackGenerator()
    
    # Generate feedback
    feedback = feedback_generator.generate_feedback(analysis)
    
    # Display feedback
    print(f"Generated {len(feedback)} feedback items:")
    for i, item in enumerate(feedback):
        print(f"\n{Fore.GREEN}Suggestion {i+1}: {item['type']}{Style.RESET_ALL}")
        print(f"  Description: {item['description']}")
        print(f"  Suggestion: {item['suggestion']}")
        print(f"  Impact: {item['impact']}")
        print(f"  Action: {item['action_mapping']['action_type']}")
    
    print_section("Apply Feedback")
    transition = StateTransition()
    
    # Map feedback to actions
    actions = feedback_generator.map_feedback_to_actions(feedback)
    
    # Apply first feedback action
    if actions:
        improved_state = transition.apply(state, actions[0])
        
        print(f"Applied action: {actions[0]}")
        print("\nImproved State:")
        print_state(improved_state)
        
        # Calculate improvement
        reward_fn = RewardFunction(
            task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
        )
        
        initial_reward = reward_fn.calculate(state)
        improved_reward = reward_fn.calculate(improved_state)
        improvement = improved_reward - initial_reward
        
        print(f"\nImprovement: {improvement:.4f} ({initial_reward:.4f} → {improved_reward:.4f})")

def demo_full_optimization():
    """Demonstrate a full optimization process with all components."""
    print_header("Full Optimization Process Demonstration")
    
    # Create initial state
    basic_text = """
    Task: Analyze the sentiment of the text.
    """
    
    state = PromptState(basic_text)
    transition = StateTransition()
    reward_fn = RewardFunction(
        task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
    )
    
    print_section("Initial State")
    print_state(state)
    
    print_section("Running MCTS Optimization with Evolutionary Features")
    # Create MCTS engine with full configuration
    mcts_engine = MCTSEngine(
        transition=transition,
        reward_function=reward_fn,
        max_iterations=50,
        time_limit=10.0,
        exploration_weight=1.41,
        max_children_per_expansion=3,
        evolution_config={
            "mutation_rate": 0.2,
            "crossover_rate": 0.2,
            "error_feedback_rate": 0.6,
            "adaptive_adjustment": True
        }
    )
    
    # Run optimization
    print("Optimizing prompt...")
    best_state, stats = mcts_engine.optimize(state)
    
    print_section("Optimization Statistics")
    print(f"Iterations: {stats['iterations']}")
    print(f"Time: {stats['time']:.2f} seconds")
    print(f"Tree size: {stats['tree_size']} nodes")
    print(f"Max depth: {stats['max_depth']}")
    print(f"Evolutionary operations: {stats['evolutionary_operations']}")
    print(f"  Mutations: {stats['mutations']}")
    print(f"  Crossovers: {stats['crossovers']}")
    print(f"Error feedback actions: {stats['error_feedback_actions']}")
    
    # 添加可视化部分
    print_section("MCTS 树结构")
    root = mcts_engine._get_root_node()
    if root:
        print_mcts_tree(root)
        
        # 收集所有节点的奖励
        all_rewards = []
        def collect_rewards(node):
            if node.visit_count > 0:
                all_rewards.append(node.avg_reward)
            for child in node.children.values():
                collect_rewards(child)
                
        collect_rewards(root)
        print_reward_histogram(all_rewards)
    
    # 打印操作统计
    print_operation_stats(stats)
    
    print_section("Best Prompt Found")
    print_state(best_state)
    
    # Compare with initial state
    initial_reward = reward_fn.calculate(state)
    best_reward = reward_fn.calculate(best_state)
    improvement = best_reward - initial_reward
    
    print(f"\nImprovement: {improvement:.4f} ({initial_reward:.4f} → {best_reward:.4f})")
    
    print_section("Final Best Prompt")
    print(f"{Fore.GREEN}{best_state.text.strip()}{Style.RESET_ALL}")

def main():
    """Main function"""
    print_header("MCTS Strategic Planning with Evolutionary Algorithms Demo")
    
    menu_options = [
        "MCTS Node Demonstration",
        "Selection and Expansion Demonstration",
        "Simulation and Backpropagation Demonstration",
        "MCTS Engine Demonstration",
        "Evolutionary Operations Demonstration",
        "Error Feedback System Demonstration",
        "Full Optimization Process Demonstration",
        "Exit"
    ]
    
    while True:
        print_section("Function Options")
        for i, option in enumerate(menu_options, 1):
            print(f"{i}. {option}")
        
        try:
            choice = input(f"\n{Fore.CYAN}Please select a function (1-{len(menu_options)}): {Style.RESET_ALL}")
            choice = int(choice)
            
            if choice == 1:
                demo_mcts_node()
            elif choice == 2:
                demo_selection_expansion()
            elif choice == 3:
                demo_simulation_backpropagation()
            elif choice == 4:
                demo_mcts_engine()
            elif choice == 5:
                demo_evolutionary_operations()
            elif choice == 6:
                demo_error_feedback()
            elif choice == 7:
                demo_full_optimization()
            elif choice == 8:
                print("\nThank you for using the MCTS Strategic Planning Demo!\n")
                break
            else:
                print(f"{Fore.RED}Invalid option, please try again{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print("\n\nProgram interrupted")
            break
        except Exception as e:
            print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()