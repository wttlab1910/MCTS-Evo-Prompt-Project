"""
Complete system test for the MCTS-Evo-Prompt project.
"""
import argparse
import asyncio
import time
import os
from pathlib import Path
import json
import base64
from datetime import datetime
import re

from app.core.mdp.state import PromptState
from app.core.mdp.action import create_action
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction

from app.core.mcts.engine import MCTSEngine
from app.core.optimization.prompt_selector import PromptSelector
from app.core.optimization.token_optimizer import TokenOptimizer
from app.core.optimization.output_processor import OutputProcessor

from app.services.output_service import OutputService
from app.utils.visualization import OptimizationVisualizer

from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider
from app.llm.evaluation.async_evaluator import AsyncPromptEvaluator

class SystemTest:
    """
    Comprehensive system test for MCTS-Evo-Prompt.
    """
    
    def __init__(self):
        """Initialize the system test."""
        # Register Ollama provider
        LLMFactory.register_provider("ollama", OllamaProvider)
        
        # Create output directory
        self.output_dir = Path("./system_test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test cases
        self.test_cases = [
            {
                "name": "sentiment_analysis",
                "prompt": "Analyze the sentiment of the text.",
                "data": "I really enjoyed the movie. The acting was superb and the plot was engaging.",
                "task_type": "sentiment_analysis"
            },
            {
                "name": "classification",
                "prompt": "Classify the text into one of these categories: Technology, Sports, Politics, Entertainment.",
                "data": "Apple announced its new iPhone with improved camera and processor.",
                "task_type": "classification"
            },
            {
                "name": "summarization",
                "prompt": "Summarize the text.",
                "data": "Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data. They are used in a variety of applications in computer vision, natural language processing, and playing games. Neural networks have seen a resurgence due to advancements in computational power and techniques for training deeper networks.",
                "task_type": "summarization"
            }
        ]
        
        print(f"Initialized SystemTest with {len(self.test_cases)} test cases")
    
    async def run_all_tests(self, model_id: str = "mistral"):
        """
        Run all test cases.
        
        Args:
            model_id: Ollama model ID to use.
        """
        print(f"\n{'='*80}")
        print(f"Starting system test with model: {model_id}")
        print(f"{'='*80}\n")
        
        # Set up LLM
        llm = LLMFactory.create("ollama", model_id=model_id)
        
        # Set up evaluator
        evaluator = AsyncPromptEvaluator(llm)
        await evaluator.initialize()
        
        # Set up sync evaluator
        sync_evaluator = self._create_sync_evaluator(evaluator)
        
        # Create components
        transition = StateTransition()
        reward_fn = RewardFunction(
            task_performance_fn=sync_evaluator,
            structural_weight=0.3,
            efficiency_weight=0.1
        )
        
        # Set up output service
        output_service = OutputService()
        visualizer = OptimizationVisualizer()
        
        # Results tracking
        results = []
        
        # Run each test case
        for test_case in self.test_cases:
            print(f"\n{'='*60}")
            print(f"Running test case: {test_case['name']}")
            print(f"Task type: {test_case['task_type']}")
            print(f"{'='*60}\n")
            
            # Create initial state
            initial_state = PromptState(test_case["prompt"])
            
            # Create MCTS engine
            mcts_engine = MCTSEngine(
                transition=transition,
                reward_function=reward_fn,
                max_iterations=30,  # Reduced for testing
                time_limit=120.0,
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
            print(f"Optimizing prompt: {test_case['prompt']}")
            start_time = time.time()
            best_state, stats = mcts_engine.optimize(initial_state)
            elapsed = time.time() - start_time
            
            print(f"Optimization completed in {elapsed:.2f}s")
            print(f"Iterations: {stats['iterations']}")
            print(f"Tree size: {stats['tree_size']} nodes")
            print(f"Best reward: {stats['best_reward']:.4f}")
            
            # Process output
            root_node = mcts_engine._root_node
            output_result = output_service.generate_output(
                root_node=root_node,
                original_data=test_case["data"],
                selection_strategy="composite",
                verification_level="standard"
            )
            
            # Compare with original
            comparison = output_service.compare_with_original(
                original_prompt=test_case["prompt"],
                optimized_prompt=best_state.text
            )
            
            # Generate visualizations
            tree_viz = visualizer.generate_tree_visualization(root_node)
            trajectory_viz = visualizer.generate_trajectory_visualization(
                output_service.prompt_selector.analyze_trajectories(root_node)
            )
            reward_viz = visualizer.generate_reward_progression_visualization(best_state)
            component_viz = visualizer.generate_component_comparison_visualization(
                comparison["original_components"],
                comparison["optimized_components"]
            )
            
            # Save visualizations
            case_dir = self.output_dir / test_case["name"]
            case_dir.mkdir(exist_ok=True)
            
            if tree_viz:
                with open(case_dir / "tree_visualization.png", "wb") as f:
                    f.write(base64.b64decode(tree_viz))
            
            if trajectory_viz:
                with open(case_dir / "trajectory_visualization.png", "wb") as f:
                    f.write(base64.b64decode(trajectory_viz))
                    
            if reward_viz:
                with open(case_dir / "reward_progression.png", "wb") as f:
                    f.write(base64.b64decode(reward_viz))
                    
            if component_viz:
                with open(case_dir / "component_comparison.png", "wb") as f:
                    f.write(base64.b64decode(component_viz))
            
            # Test with LLM
            response = await self._test_with_llm(
                llm, 
                output_result["final_output"], 
                test_case["data"]
            )
            
            # Evaluate the optimized prompt
            eval_result = await self._evaluate_with_llm(
                llm,
                output_result["final_output"],
                test_case["prompt"]
            )
            
            # Save the complete test result
            test_result = {
                "name": test_case["name"],
                "task_type": test_case["task_type"],
                "original_prompt": test_case["prompt"],
                "data": test_case["data"],
                "optimized_prompt": best_state.text,
                "final_output": output_result["final_output"],
                "optimization_stats": stats,
                "processing_stats": output_result["processing_stats"],
                "comparison": comparison,
                "llm_response": response.get("text", ""),
                "evaluation": eval_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save the test result
            with open(case_dir / "result.json", "w") as f:
                json.dump(test_result, f, indent=2)
            
            # Add to overall results
            results.append(test_result)
            
            # Print results
            print("\nOriginal prompt:")
            print(test_case["prompt"])
            print("\nOptimized prompt:")
            print(best_state.text)
            print("\nFinal output:")
            print(output_result["final_output"])
            print("\nLLM response:")
            print(response.get("text", "")[:500] + "..." if len(response.get("text", "")) > 500 else response.get("text", ""))
            print("\nEvaluation:")
            print(eval_result.get("evaluation_text", "")[:500] + "..." if len(eval_result.get("evaluation_text", "")) > 500 else eval_result.get("evaluation_text", ""))
        
        # Save overall results
        with open(self.output_dir / "all_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate summary
        self._generate_summary(results)
        
        print(f"\n{'='*80}")
        print(f"System test completed. Results saved to {self.output_dir}")
        print(f"{'='*80}\n")
    
    def _create_sync_evaluator(self, async_evaluator):
        """Create a synchronous wrapper for the async evaluator."""
        class SyncEvaluator:
            def __init__(self, async_eval):
                self.async_eval = async_eval
                
            def __call__(self, state, data=None):
                # Run the async evaluator in the current event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.async_eval.evaluate_prompt(state, data))
        
        return SyncEvaluator(async_evaluator)
    
    async def _test_with_llm(self, llm, prompt, data=None):
        """Test a prompt with the LLM."""
        full_prompt = prompt
        if data:
            full_prompt = f"{prompt}\n\n{data}"
        
        try:
            return await llm.generate(full_prompt)
        except Exception as e:
            print(f"Error testing with LLM: {e}")
            return {"text": f"Error: {str(e)}"}
    
    async def _evaluate_with_llm(self, llm, prompt, original_prompt):
        """Evaluate a prompt using the LLM."""
        evaluation_prompt = f"""
Analyze the quality of the following optimized prompt compared to the original prompt.

Original Prompt:
{original_prompt}

Optimized Prompt:
{prompt}

Provide an evaluation covering:
1. Clarity and structure improvements
2. Information completeness 
3. Guidance effectiveness
4. Potential improvements

Then rate the optimized prompt on a scale of 1-10 where 10 is excellent.
        """
        
        try:
            response = await llm.generate(evaluation_prompt)
            
            # Extract the numerical rating if present
            text = response.get("text", "")
            rating = None
            
            # Try to find a numerical rating
            rating_match = re.search(r'(\d+(\.\d+)?)/10', text)
            if rating_match:
                rating = float(rating_match.group(1))
            else:
                # Try to find any number that looks like a rating
                number_match = re.search(r'rating:\s*(\d+(\.\d+)?)', text, re.IGNORECASE)
                if number_match:
                    rating = float(number_match.group(1))
            
            return {
                "evaluation_text": text,
                "rating": rating,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Error evaluating with LLM: {e}")
            return {
                "evaluation_text": f"Error: {str(e)}",
                "rating": None,
                "timestamp": time.time()
            }
    
    def _generate_summary(self, results):
        """Generate a summary of test results."""
        summary = {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r.get("evaluation", {}).get("rating", 0) >= 7),
            "average_rating": sum(r.get("evaluation", {}).get("rating", 0) for r in results if r.get("evaluation", {}).get("rating")) / 
                             sum(1 for r in results if r.get("evaluation", {}).get("rating")),
            "average_component_increase": sum(r["comparison"]["components_added"] for r in results) / len(results),
            "improvement_categories": {}
        }
        
        # Count improvement categories
        all_improvements = []
        for r in results:
            all_improvements.extend(r["comparison"]["structural_improvements"])
        
        for improvement in all_improvements:
            category = improvement.split(":")[0] if ":" in improvement else improvement
            summary["improvement_categories"][category] = summary["improvement_categories"].get(category, 0) + 1
        
        # Save summary
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\nTest Summary:")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful tests: {summary['successful_tests']}")
        print(f"Average rating: {summary['average_rating']:.2f}/10")
        print(f"Average component increase: {summary['average_component_increase']:.2f}")
        print("Improvement categories:")
        for category, count in summary["improvement_categories"].items():
            print(f"  - {category}: {count}")

def main():
    """Run the system test."""
    parser = argparse.ArgumentParser(description="MCTS-Evo-Prompt System Test")
    parser.add_argument("--model", default="mistral", help="Ollama model ID")
    
    args = parser.parse_args()
    
    # Run the system test
    test = SystemTest()
    asyncio.run(test.run_all_tests(model_id=args.model))

if __name__ == "__main__":
    main()