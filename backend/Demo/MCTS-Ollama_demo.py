"""
MCTS-Ollama Demo - Demonstrates prompt optimization using MCTS with Ollama LLM integration.

This demo provides a simplified interface to test the MCTS algorithm with Ollama models.
"""
import sys
import os
import asyncio
import time
import random
from pathlib import Path
import argparse
from colorama import init, Fore, Style

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import MCTS components
from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction

from app.core.mcts.node import MCTSNode
from app.core.mcts.engine import MCTSEngine

# Import Ollama LLM
from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider

# Initialize colorama
init()

# Register Ollama provider
LLMFactory.register_provider("ollama", OllamaProvider)

class OllamaEvaluator:
    """Custom evaluator using Ollama LLM."""
    
    def __init__(self, model_name="mistral", ctx_size=1024, max_tokens=100):
        """Initialize the Ollama evaluator."""
        self.model_name = model_name
        self.ctx_size = ctx_size
        self.max_tokens = max_tokens
        self.llm = None
        
    async def initialize(self):
        """Initialize the LLM."""
        if not self.llm:
            self.llm = LLMFactory.create(
                "ollama", 
                model_id=self.model_name,
                ctx_size=self.ctx_size
            )
            
    async def evaluate(self, prompt_state: PromptState, data: str = None) -> float:
        """
        Evaluate a prompt state using the Ollama LLM.
        
        Args:
            prompt_state: The prompt state to evaluate.
            data: Optional data to include in the prompt.
            
        Returns:
            Reward value between 0 and 1.
        """
        await self.initialize()
        
        # Combine prompt with data if provided
        full_prompt = prompt_state.text.strip()
        if data:
            full_prompt += f"\n\n{data}"
            
        # Add evaluation instructions with stronger directive for differentiation
        eval_prompt = f"""
{full_prompt}

On a scale of 0 to 10, rate the quality of the above prompt for sentiment analysis.
Consider clarity, completeness, and structure.
Be critical in your assessment - simple prompts should score lower than detailed, well-structured ones.
Just provide a single number between 0 and 10.
        """.strip()
        
        try:
            # Generate response with limited tokens
            response = await self.llm.generate(
                eval_prompt, 
                max_tokens=self.max_tokens
            )
            
            # Extract the rating
            text = response.get("text", "").strip()
            
            # Try to find a number in the response
            rating = None
            for word in text.split():
                try:
                    rating = float(word.replace(",", "."))
                    if 0 <= rating <= 10:
                        break
                except ValueError:
                    continue
            
            # Default to mid-range if no valid rating found
            if rating is None:
                print(f"Warning: Could not extract rating from response: {text}")
                return 0.5
                
            # Normalize to 0-1 range
            return rating / 10.0
            
        except Exception as e:
            print(f"Error evaluating with Ollama: {e}")
            return 0.5  # Default mid-range reward

# Synchronous wrapper for the async evaluator
class SyncOllamaEvaluator:
    """Synchronous wrapper for OllamaEvaluator."""
    
    def __init__(self, async_evaluator):
        """Initialize with an async evaluator instance."""
        self.async_evaluator = async_evaluator
        
    def __call__(self, prompt_state: PromptState, data: str = None) -> float:
        """
        Synchronous evaluate function that can be called directly.
        
        Args:
            prompt_state: The prompt state to evaluate.
            data: Optional data to include in the prompt.
            
        Returns:
            Reward value between 0 and 1.
        """
        # Check if we're already inside an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an event loop, we need to use a different approach
                # We create a future and wait for it
                future = asyncio.run_coroutine_threadsafe(
                    self.async_evaluator.evaluate(prompt_state, data), 
                    loop
                )
                return future.result()
        except RuntimeError:
            # No running event loop, create one
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.async_evaluator.evaluate(prompt_state, data))
            finally:
                loop.close()

def print_header(text):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 80}")
    print(f"{text.center(80)}")
    print(f"{'=' * 80}{Style.RESET_ALL}")

def print_section(text):
    """Print colored section title"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_state(state, reward=None):
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
    
    # Print reward if provided
    if reward is not None:
        print(f"\n{Fore.CYAN}Reward: {reward:.4f}{Style.RESET_ALL}")

def print_mcts_tree(root, max_depth=2, current_depth=0):
    """Print a simplified view of the MCTS tree"""
    if current_depth > max_depth:
        print(f"{'  ' * current_depth}... (tree too deep, truncated)")
        return
        
    # Print node information
    if root.visit_count > 0:
        if root.avg_reward > 0.7:
            color = Fore.GREEN
        elif root.avg_reward > 0.4:
            color = Fore.YELLOW
        else:
            color = Fore.RED
    else:
        color = Fore.WHITE
    
    indent = "  " * current_depth
    print(f"{indent}{color}Node {root.node_id[:8]}: visits={root.visit_count}, "
          f"avg_reward={root.avg_reward:.4f}, children={len(root.children)}{Style.RESET_ALL}")
    
    # Print action if available
    if root.action_applied:
        print(f"{indent}  Action: {Fore.CYAN}{root.action_applied}{Style.RESET_ALL}")
    
    # Only display a few children to avoid overwhelming output
    if root.children:
        sorted_children = sorted(
            root.children.values(), 
            key=lambda x: x.avg_reward if x.visit_count > 0 else 0, 
            reverse=True
        )
        
        # Print top 3 children
        for i, child in enumerate(sorted_children[:3]):
            print_mcts_tree(child, max_depth, current_depth + 1)
            
        if len(sorted_children) > 3:
            print(f"{indent}  ... ({len(sorted_children) - 3} more children)")

def print_operation_stats(stats):
    """Print operation statistics"""
    print("\nOperation Statistics:")
    
    # Calculate total operations
    total_ops = sum([
        stats.get("mutations", 0),
        stats.get("crossovers", 0),
        stats.get("error_feedback_actions", 0)
    ])
    
    if total_ops == 0:
        print("No operations performed")
        return
    
    # Print statistics
    print(f"  Mutations: {stats.get('mutations', 0)} ({stats.get('mutations', 0)/total_ops*100:.1f}%)")
    print(f"  Crossovers: {stats.get('crossovers', 0)} ({stats.get('crossovers', 0)/total_ops*100:.1f}%)")
    print(f"  Error feedback: {stats.get('error_feedback_actions', 0)} ({stats.get('error_feedback_actions', 0)/total_ops*100:.1f}%)")

def print_reward_histogram(rewards, bins=10, width=40):
    """Print ASCII histogram of rewards"""
    if not rewards:
        print("No reward data available")
        return
        
    print("\nReward Distribution:")
    
    # Calculate histogram data
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    if min_reward == max_reward:
        print(f"All rewards are: {min_reward:.4f}")
        return
        
    bin_width = (max_reward - min_reward) / bins
    
    # Count rewards in each bin
    counts = [0] * bins
    for reward in rewards:
        bin_idx = min(bins - 1, int((reward - min_reward) / bin_width))
        counts[bin_idx] += 1
    
    # Find maximum count for scaling
    max_count = max(counts) if counts else 0
    scale = width / max_count if max_count > 0 else 1
    
    # Print histogram
    for i in range(bins):
        bin_min = min_reward + i * bin_width
        bin_max = bin_min + bin_width
        bar_length = int(counts[i] * scale)
        bar = "#" * bar_length
        print(f"{bin_min:.2f}-{bin_max:.2f} [{counts[i]:3d}]: {bar}")

async def collect_all_rewards(root):
    """Collect all rewards from the MCTS tree"""
    rewards = []
    
    def collect(node):
        if node.visit_count > 0:
            rewards.append(node.avg_reward)
        for child in node.children.values():
            collect(child)
    
    collect(root)
    return rewards

async def optimize_prompt(model_name="mistral", iterations=30, time_limit=60.0, initial_prompt=None, data=None):
    """Run MCTS optimization with Ollama evaluator"""
    print_header(f"MCTS Prompt Optimization with {model_name.upper()}")
    
    # Create initial state
    if initial_prompt is None:
        initial_prompt = """
        Task: Analyze the sentiment of the text.
        """
    
    print_section("Initial Prompt")
    state = PromptState(initial_prompt)
    print_state(state)
    
    # Set up MCTS components
    transition = StateTransition()
    
    # Create and initialize Ollama evaluator
    async_evaluator = OllamaEvaluator(model_name=model_name, ctx_size=1024, max_tokens=100)
    await async_evaluator.initialize()
    
    # Create a synchronous wrapper for the asynchronous evaluator
    sync_evaluator = SyncOllamaEvaluator(async_evaluator)
    
    # Create reward function with the synchronous wrapper
    reward_fn = RewardFunction(
        task_performance_fn=sync_evaluator,  # Use the synchronous wrapper
        structural_weight=0.3,
        efficiency_weight=0.1 
    )
    
    # Get initial reward - need to await the async evaluation
    print("\nEvaluating initial prompt...")
    initial_reward = await async_evaluator.evaluate(state, data)
    print(f"Initial reward: {initial_reward:.4f}")
    
    # Create MCTS engine
    mcts_engine = MCTSEngine(
        transition=transition,
        reward_function=reward_fn,
        max_iterations=iterations,
        time_limit=time_limit,
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
    print_section("Running MCTS Optimization")
    print(f"Using {model_name} model with {iterations} iterations and {time_limit}s time limit")
    print("\nOptimizing prompt... (this may take a while)")
    
    start_time = time.time()
    best_state, stats = mcts_engine.optimize(state)
    elapsed = time.time() - start_time
    
    # Print statistics
    print_section("Optimization Results")
    print(f"Optimization completed in {elapsed:.2f}s")
    print(f"Iterations: {stats['iterations']}")
    print(f"Tree size: {stats['tree_size']} nodes")
    print(f"Max depth: {stats['max_depth']}")
    print(f"Best reward: {stats['best_reward']:.4f}")
    
    # Print operation statistics
    print_operation_stats(stats)
    
    # Print MCTS tree structure
    print_section("MCTS Tree Structure")
    root = mcts_engine._get_root_node()
    if root:
        print_mcts_tree(root)
        
        # Collect and print reward distribution
        rewards = await collect_all_rewards(root)
        print_reward_histogram(rewards)
    
    # Print best prompt
    print_section("Optimized Prompt")
    # Use the async evaluator directly here since we're in an async function
    best_reward = await async_evaluator.evaluate(best_state, data)
    print_state(best_state, best_reward)
    
    # Calculate improvement
    improvement = best_reward - initial_reward
    print(f"\nImprovement: {Fore.GREEN}{improvement:.4f}{Style.RESET_ALL} ({initial_reward:.4f} → {best_reward:.4f})")
    
    # Return results
    return {
        "initial_state": state,
        "initial_reward": initial_reward,
        "best_state": best_state,
        "best_reward": best_reward,
        "improvement": improvement,
        "stats": stats,
        "elapsed_time": elapsed
    }

async def compare_models():
    """Compare different Ollama models for MCTS optimization"""
    print_header("Model Comparison for MCTS Optimization")
    
    # Define models to compare
    models = ["mistral", "gemma3:12b", "deepseek-r1:32b"]
    
    # Define a simple prompt and data
    prompt = """
    Task: Analyze the sentiment of the following review.
    """
    
    data = "This restaurant was amazing! The food was delicious and the service was exceptional."
    
    # Set up optimization parameters
    iterations = 10  # Keep low for quick comparison
    time_limit = 30.0
    
    # Run optimization with each model
    results = {}
    for model in models:
        try:
            print_section(f"Testing Model: {model}")
            result = await optimize_prompt(
                model_name=model,
                iterations=iterations,
                time_limit=time_limit,
                initial_prompt=prompt,
                data=data
            )
            results[model] = result
        except Exception as e:
            print(f"{Fore.RED}Error testing {model}: {e}{Style.RESET_ALL}")
    
    # Compare results
    print_section("Comparison Summary")
    
    # Create comparison table
    print("\nModel Comparison:")
    print("-" * 80)
    print(f"{'Model':20} {'Initial':10} {'Final':10} {'Improvement':12} {'Time (s)':10} {'Iterations':10}")
    print("-" * 80)
    
    for model, result in results.items():
        print(f"{model:20} {result['initial_reward']:.4f}    {result['best_reward']:.4f}    " + 
              f"{result['improvement']:.4f}       {result['elapsed_time']:.2f}      " +
              f"{result['stats']['iterations']}")
    
    print("-" * 80)
    
    # Determine best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['improvement'])[0]
        print(f"\nBest performing model: {Fore.GREEN}{best_model}{Style.RESET_ALL}")
        print(f"Achieved improvement: {results[best_model]['improvement']:.4f}")

async def test_complex_task():
    """Test MCTS optimization on a more complex task"""
    print_header("Complex Task Optimization")
    
    # Complex task: Text classification prompt
    complex_prompt = """
    Task: Classify the given text into one of the following categories: 
    Business, Technology, Sports, Politics, Entertainment.
    """
    
    data = """
    Apple Inc. has announced a new AI strategy that will integrate advanced 
    machine learning capabilities across all its product lines. The company plans
    to unveil these features at their upcoming developer conference.
    """
    
    # Run optimization
    await optimize_prompt(
        model_name="mistral",  # Use your fastest model
        iterations=20,
        time_limit=45.0,
        initial_prompt=complex_prompt,
        data=data
    )

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MCTS-Ollama Demo")
    parser.add_argument("--task", type=str, default="basic", 
                      choices=["basic", "compare", "complex"],
                      help="Task to perform (basic, compare, complex)")
    parser.add_argument("--model", type=str, default="mistral",
                      help="Model to use (mistral, gemma3:12b, deepseek-r1:32b)")
    parser.add_argument("--iterations", type=int, default=30,
                      help="Number of MCTS iterations")
    parser.add_argument("--time", type=float, default=60.0,
                      help="Time limit in seconds")
    
    args = parser.parse_args()
    
    try:
        if args.task == "basic":
            await optimize_prompt(
                model_name=args.model,
                iterations=args.iterations,
                time_limit=args.time
            )
        elif args.task == "compare":
            await compare_models()
        elif args.task == "complex":
            await test_complex_task()
        else:
            print(f"Unknown task: {args.task}")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())