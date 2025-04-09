"""
Comprehensive integration test for Phase 5: Final Prompt Generation and Output.
This test interacts with a real LLM (Ollama) to validate the complete workflow.
"""
import asyncio
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import base64
# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from app.core.mdp.state import PromptState
from app.core.mcts.node import MCTSNode
from app.core.optimization.prompt_selector import PromptSelector
from app.core.optimization.token_optimizer import TokenOptimizer
from app.core.optimization.output_processor import OutputProcessor
from app.services.output_service import OutputService
from app.utils.visualization import OptimizationVisualizer

from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider
from app.llm.evaluation.async_evaluator import AsyncPromptEvaluator

class TestPhase5Integration:
    """
    Integration tests for Phase 5 with real LLM interaction.
    """
    
    @classmethod
    async def setup_class(cls):
        """Set up test fixtures for the entire test class."""
        print("\n" + "="*80)
        print("PHASE 5 INTEGRATION TEST WITH REAL LLM")
        print("="*80)
        
        # Register Ollama provider
        LLMFactory.register_provider("ollama", OllamaProvider)
        
        # Create LLM instance
        try:
            cls.llm = LLMFactory.create("ollama", model_id="mistral")
            cls.llm_available = True
            print("✅ Successfully connected to Ollama LLM service")
        except Exception as e:
            cls.llm_available = False
            print(f"❌ Failed to connect to Ollama LLM service: {e}")
            print("⚠️ Some tests will be skipped or limited")
        
        # Create evaluator if LLM is available
        if cls.llm_available:
            cls.evaluator = AsyncPromptEvaluator(cls.llm)
            await cls.evaluator.initialize()
        
        # Create output directory
        cls.output_dir = Path("./test_phase5_output")
        cls.output_dir.mkdir(exist_ok=True)
        
        # Create sample prompt states
        cls.basic_state = PromptState(
            "Analyze the sentiment of the text."
        )
        
        cls.structured_state = PromptState(
            """
            Role: Sentiment Analysis Expert
            Task: Analyze the sentiment of the provided text.
            
            Steps:
            - Read the text carefully
            - Identify sentiment-bearing words and phrases
            - Determine overall sentiment
            
            Output Format: Provide sentiment as positive, negative, or neutral.
            """
        )
        
        cls.sample_data = "I really enjoyed this movie. The acting was superb and the plot was engaging."
        
        # Create a mock MCTS tree
        cls.root_node = MCTSNode(state=cls.basic_state)
        cls.root_node.update_statistics(0.5)
        
        child1_state = PromptState("Role: Analyst\nTask: Analyze the sentiment of the text.")
        cls.child1 = cls.root_node.add_child(None, child1_state)
        cls.child1.update_statistics(0.6)
        
        child2_state = PromptState(cls.structured_state.text)
        cls.child2 = cls.root_node.add_child(None, child2_state)
        cls.child2.update_statistics(0.8)
        cls.child2.update_statistics(0.9)
        
        # Create components
        cls.prompt_selector = PromptSelector()
        cls.token_optimizer = TokenOptimizer()
        cls.output_processor = OutputProcessor(token_optimizer=cls.token_optimizer)
        cls.output_service = OutputService()
        cls.visualizer = OptimizationVisualizer()
        
        print("✅ Test setup completed")
    
    @classmethod
    async def teardown_class(cls):
        """Clean up after tests."""
        print("\n" + "="*80)
        print("TEST CLEANUP")
        print("="*80)
        print(f"Test output saved to: {cls.output_dir}")
        print("✅ Test cleanup completed")
    
    async def test_01_prompt_selector(self):
        """Test prompt selector with various strategies."""
        print("\n" + "-"*80)
        print("TEST 1: PROMPT SELECTOR STRATEGIES")
        print("-"*80)
        
        # Test with different strategies
        strategies = ["global_max", "path_max", "composite", "stable"]
        
        for strategy in strategies:
            # Select optimal prompt
            best_state, stats = self.prompt_selector.select_optimal_prompt(
                self.root_node, strategy=strategy
            )
            
            print(f"\nStrategy: {strategy}")
            print(f"- Best reward: {stats['best_reward']:.4f}")
            print(f"- Nodes evaluated: {stats.get('nodes_evaluated', 'N/A')}")
            
            # Verify selection worked
            assert best_state is not None
            assert stats["strategy"] == strategy
            assert stats["best_reward"] > 0
        
        # Test trajectory analysis
        trajectories = self.prompt_selector.analyze_trajectories(self.root_node, top_k=2)
        
        print("\nTrajectory Analysis:")
        for i, trajectory in enumerate(trajectories):
            print(f"- Trajectory {i+1}: Score = {trajectory['evaluation']['path_score']:.4f}, "
                 f"Length = {trajectory['evaluation']['path_length']}")
        
        # Generate visualization
        viz_data = self.visualizer.generate_trajectory_visualization(trajectories)
        if viz_data:
            with open(self.output_dir / "trajectories.png", "wb") as f:
                f.write(base64.b64decode(viz_data))
            print("✅ Saved trajectory visualization")
        
        print("✅ Prompt selector test passed")
    
    async def test_02_token_optimizer(self):
        """Test token optimizer functionality."""
        print("\n" + "-"*80)
        print("TEST 2: TOKEN OPTIMIZER")
        print("-"*80)
        
        # Create text with filler words and verbose phrases
        verbose_state = PromptState(
            """
            Role: Data Analyst
            Task: In order to analyze the sentiment of the text that is provided, you should generally process the input.
            
            Steps:
            - At the present time, read the text carefully and thoroughly
            - It is important to note that you should identify sentiment words
            - Due to the fact that sentiment can be positive or negative, classify accordingly
            
            Output Format: Provide sentiment as positive, negative, or neutral.
            """
        )
        
        print("Original text:")
        print(verbose_state.text)
        print(f"Original length: {len(verbose_state.text)} characters")
        
        # Test standard optimization
        optimized_state, stats = self.token_optimizer.optimize(verbose_state)
        
        print("\nOptimized text (standard mode):")
        print(optimized_state.text)
        print(f"Optimized length: {len(optimized_state.text)} characters")
        print(f"Characters reduced: {stats['characters_reduced']} ({stats['reduction_percent']:.2f}%)")
        print(f"Filler words removed: {stats['filler_words_removed']}")
        print(f"Verbose phrases replaced: {stats['phrases_replaced']}")
        
        # Test aggressive optimization
        aggressive_optimizer = TokenOptimizer(aggressive_mode=True)
        aggressive_state, aggressive_stats = aggressive_optimizer.optimize(verbose_state)
        
        print("\nOptimized text (aggressive mode):")
        print(aggressive_state.text)
        print(f"Optimized length: {len(aggressive_state.text)} characters")
        print(f"Characters reduced: {aggressive_stats['characters_reduced']} ({aggressive_stats['reduction_percent']:.2f}%)")
        
        # Check optimization effectiveness
        assert len(optimized_state.text) < len(verbose_state.text)
        assert "In order to" not in optimized_state.text
        assert "Due to the fact that" not in optimized_state.text
        
        if aggressive_stats['reduction_percent'] > stats['reduction_percent']:
            print("✅ Aggressive mode provides additional optimization")
        else:
            print("⚠️ Aggressive mode did not provide significant additional optimization for this example")
        
        print("✅ Token optimizer test passed")
    
    async def test_03_output_processor(self):
        """Test output processor functionality."""
        print("\n" + "-"*80)
        print("TEST 3: OUTPUT PROCESSOR")
        print("-"*80)
        
        # Test verification with missing components
        incomplete_state = PromptState("Analyze the sentiment.")
        
        print("Processing incomplete prompt:")
        print(incomplete_state.text)
        
        # Process with minimal verification
        output_minimal, stats_minimal = self.output_processor.process_output(
            incomplete_state, 
            verification_level="minimal"
        )
        
        print("\nWith minimal verification:")
        print(output_minimal)
        print(f"Verification passed: {stats_minimal['verification_passed']}")
        print(f"Issues found: {len(stats_minimal['verification_issues'])}")
        
        # Process with standard verification
        output_standard, stats_standard = self.output_processor.process_output(
            incomplete_state, 
            verification_level="standard"
        )
        
        print("\nWith standard verification:")
        print(output_standard)
        print(f"Verification passed: {stats_standard['verification_passed']}")
        print(f"Issues found: {len(stats_standard['verification_issues'])}")
        
        # Process with thorough verification
        output_thorough, stats_thorough = self.output_processor.process_output(
            incomplete_state, 
            verification_level="thorough"
        )
        
        print("\nWith thorough verification:")
        print(output_thorough)
        print(f"Verification passed: {stats_thorough['verification_passed']}")
        print(f"Issues found: {len(stats_thorough['verification_issues'])}")
        
        # Test with data integration
        output_with_data, stats_with_data = self.output_processor.process_output(
            self.structured_state,
            original_data=self.sample_data,
            verification_level="standard"
        )
        
        print("\nProcessing with data integration:")
        print(output_with_data)
        
        # Verify data integration
        assert self.sample_data in output_with_data
        
        # Check that more thorough verification finds more issues
        assert len(stats_thorough['verification_issues']) >= len(stats_standard['verification_issues'])
        assert len(stats_standard['verification_issues']) >= len(stats_minimal['verification_issues'])
        
        print("✅ Output processor test passed")
    
    async def test_04_output_service(self):
        """Test output service full integration."""
        print("\n" + "-"*80)
        print("TEST 4: OUTPUT SERVICE INTEGRATION")
        print("-"*80)
        
        # Generate output
        result = self.output_service.generate_output(
            self.root_node,
            original_data=self.sample_data,
            selection_strategy="composite",
            verification_level="standard"
        )
        
        print("Output service results:")
        print(f"- Selection strategy: {result['selection_stats']['strategy']}")
        print(f"- Best reward: {result['selection_stats']['best_reward']:.4f}")
        print(f"- Final output length: {len(result['final_output'])}")
        print(f"- Verification passed: {result['processing_stats']['verification_passed']}")
        
        # Compare with original
        comparison = self.output_service.compare_with_original(
            original_prompt=self.basic_state.text,
            optimized_prompt=self.structured_state.text
        )
        
        print("\nComparison with original:")
        print(f"- Original length: {comparison['original_length']} chars")
        print(f"- Optimized length: {comparison['optimized_length']} chars")
        print(f"- Length difference: {comparison['length_difference']} chars")
        print(f"- Components added: {comparison['components_added']}")
        
        print("\nStructural improvements:")
        for improvement in comparison['structural_improvements']:
            print(f"- {improvement}")
        
        # Generate component comparison visualization
        viz_data = self.visualizer.generate_component_comparison_visualization(
            comparison["original_components"],
            comparison["optimized_components"]
        )
        
        if viz_data:
            with open(self.output_dir / "component_comparison.png", "wb") as f:
                f.write(base64.b64decode(viz_data))
            print("✅ Saved component comparison visualization")
        
        # Check result structure
        assert "final_output" in result
        assert "best_state" in result
        assert "selection_stats" in result
        assert "processing_stats" in result
        assert "top_trajectories" in result
        
        print("✅ Output service test passed")
    
    async def test_05_llm_evaluation(self):
        """Test evaluation with real LLM."""
        print("\n" + "-"*80)
        print("TEST 5: LLM EVALUATION")
        print("-"*80)
        
        if not self.llm_available:
            print("⚠️ LLM not available, skipping test")
            return
        
        # Evaluate original and optimized prompts
        original_prompt = self.basic_state.text
        optimized_prompt = self.structured_state.text
        
        print("Evaluating prompts with real LLM...")
        print(f"Original prompt: {original_prompt}")
        print(f"Optimized prompt: {optimized_prompt}")
        
        # Create evaluation prompt
        evaluation_prompt = f"""
Analyze the quality of the following optimized prompt compared to the original prompt.

Original Prompt:
{original_prompt}

Optimized Prompt:
{optimized_prompt}

Provide an evaluation covering:
1. Clarity and structure improvements
2. Information completeness 
3. Guidance effectiveness
4. Potential improvements

Then rate the optimized prompt on a scale of 1-10 where 10 is excellent.
        """
        
        # Send to LLM
        start_time = time.time()
        response = await self.llm.generate(evaluation_prompt)
        elapsed = time.time() - start_time
        
        # Display results
        print(f"\nLLM Evaluation (took {elapsed:.2f}s):")
        print(response.get("text", "No response"))
        
        # Try to extract rating
        import re
        rating_match = re.search(r'(\d+(\.\d+)?)/10', response.get("text", ""))
        if rating_match:
            rating = float(rating_match.group(1))
            print(f"\nExtracted rating: {rating}/10")
        
        # Test prompt with data
        print("\nTesting optimized prompt with sample data...")
        
        test_prompt = f"{optimized_prompt}\n\n{self.sample_data}"
        
        start_time = time.time()
        test_response = await self.llm.generate(test_prompt)
        elapsed = time.time() - start_time
        
        print(f"\nLLM Response (took {elapsed:.2f}s):")
        print(test_response.get("text", "No response"))
        
        print("✅ LLM evaluation test completed")
    
    async def test_06_async_evaluator(self):
        """Test async evaluator with real LLM."""
        print("\n" + "-"*80)
        print("TEST 6: ASYNC PROMPT EVALUATOR")
        print("-"*80)
        
        if not self.llm_available:
            print("⚠️ LLM not available, skipping test")
            return
        
        # Create evaluator
        print("Testing AsyncPromptEvaluator...")
        
        # Evaluate different prompts
        prompts = [
            PromptState("Analyze the sentiment."),
            PromptState("Analyze the sentiment of the given text."),
            self.structured_state
        ]
        
        # Evaluate each prompt
        for i, prompt_state in enumerate(prompts):
            print(f"\nEvaluating prompt {i+1}:")
            print(prompt_state.text)
            
            # With and without data
            score_no_data = await self.evaluator.evaluate_prompt(prompt_state)
            print(f"Score without data: {score_no_data:.4f}")
            
            score_with_data = await self.evaluator.evaluate_prompt(prompt_state, self.sample_data)
            print(f"Score with data: {score_with_data:.4f}")
        
        # Test caching
        print("\nTesting cache functionality...")
        cache_test_state = PromptState("This is a test prompt for caching.")
        
        # First evaluation (cache miss)
        start_time = time.time()
        await self.evaluator.evaluate_prompt(cache_test_state)
        first_time = time.time() - start_time
        
        # Second evaluation (cache hit)
        start_time = time.time()
        await self.evaluator.evaluate_prompt(cache_test_state)
        second_time = time.time() - start_time
        
        print(f"First evaluation: {first_time:.4f}s")
        print(f"Second evaluation: {second_time:.4f}s")
        print(f"Cache statistics: {self.evaluator.cache_hits} hits, {self.evaluator.cache_misses} misses")
        
        # Cache should make second evaluation faster
        if second_time < first_time:
            print("✅ Cache improved performance")
        else:
            print("⚠️ Cache did not improve performance significantly")
        
        print("✅ Async evaluator test completed")
    
    async def test_07_visualization(self):
        """Test visualization components."""
        print("\n" + "-"*80)
        print("TEST 7: VISUALIZATIONS")
        print("-"*80)
        
        # Generate all visualizations
        # 1. Tree visualization
        tree_viz = self.visualizer.generate_tree_visualization(self.root_node)
        if tree_viz:
            with open(self.output_dir / "tree.png", "wb") as f:
                f.write(base64.b64decode(tree_viz))
            print("✅ Generated tree visualization")
        
        # 2. Reward progression visualization
        reward_viz = self.visualizer.generate_reward_progression_visualization(self.child2)
        if reward_viz:
            with open(self.output_dir / "reward_progression.png", "wb") as f:
                f.write(base64.b64decode(reward_viz))
            print("✅ Generated reward progression visualization")
        
        # 3. Component comparison
        original_components = {
            "role": False,
            "task": True,
            "steps": False,
            "output_format": False,
            "examples": False,
            "constraints": False
        }
        
        optimized_components = {
            "role": True,
            "task": True,
            "steps": True,
            "output_format": True,
            "examples": False,
            "constraints": False
        }
        
        comp_viz = self.visualizer.generate_component_comparison_visualization(
            original_components, optimized_components
        )
        if comp_viz:
            with open(self.output_dir / "components.png", "wb") as f:
                f.write(base64.b64decode(comp_viz))
            print("✅ Generated component comparison visualization")
        
        print("✅ Visualization test completed")
        
def run_tests():
    """Run all tests."""
    asyncio.run(_run_tests())

async def _run_tests():
    """Async version of run_tests."""
    test = TestPhase5Integration()
    await test.setup_class()
    
    try:
        await test.test_01_prompt_selector()
        await test.test_02_token_optimizer()
        await test.test_03_output_processor()
        await test.test_04_output_service()
        await test.test_05_llm_evaluation()
        await test.test_06_async_evaluator()
        await test.test_07_visualization()
    finally:
        await test.teardown_class()

if __name__ == "__main__":
    run_tests()