"""
Interactive demo for the MCTS-Evo-Prompt system.
"""
import argparse
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import time
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，防止可视化错误
import matplotlib.pyplot as plt
import numpy as np
import base64

# 添加backend目录到Python路径
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

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

from app.core.optimization.prompt_selector import PromptSelector
from app.core.optimization.token_optimizer import TokenOptimizer
from app.core.optimization.output_processor import OutputProcessor

from app.services.output_service import OutputService
from app.utils.visualization import OptimizationVisualizer

# LLM Interface
from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider
from app.llm.evaluation.async_evaluator import AsyncPromptEvaluator

class MCTSEvoPromptDemo:
    """
    Demonstration of the complete MCTS-Evo-Prompt system.
    """
    
    def __init__(self, llm_provider: str = "ollama", model_id: str = "mistral", use_gpu: bool = True):
        """
        Initialize the demo.
        
        Args:
            llm_provider: LLM provider to use.
            model_id: Model ID to use.
            use_gpu: Whether to use GPU acceleration.
        """
        # 设置GPU优化选项
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU
            os.environ["OMP_NUM_THREADS"] = "4"       # 设置OpenMP线程数
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 不使用GPU
        
        # Register Ollama provider
        LLMFactory.register_provider("ollama", OllamaProvider)
        
        # Set up LLM with optimized parameters
        llm_params = {
            "n_gpu_layers": -1 if use_gpu else 0,  # 使用所有可能的GPU层
            "n_ctx": 4096,                         # 更大的上下文窗口
            "f16_kv": True                         # 使用FP16以提高速度
        }
        
        self.llm = LLMFactory.create(
            llm_provider, 
            model_id=model_id,
            **llm_params
        )
        
        # Set up performance evaluator for evaluation
        self.async_evaluator = None
        
        # Initialize output components
        self.output_service = OutputService()
        self.visualizer = OptimizationVisualizer()
        
        # Save visualization output directory
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Initialized MCTSEvoPromptDemo with {llm_provider}:{model_id}, GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    
    async def setup_evaluator(self):
        """Set up the async evaluator."""
        # Create an async evaluator for the task
        self.async_evaluator = AsyncPromptEvaluator(self.llm)
        await self.async_evaluator.initialize()
    
    def create_sync_evaluator(self):
        """Create a synchronous wrapper for the async evaluator."""
        if not self.async_evaluator:
            raise ValueError("Async evaluator not initialized. Call setup_evaluator first.")
        
        # Create a sync wrapper for the async evaluator
        class SyncEvaluator:
            def __init__(self, async_eval):
                self.async_eval = async_eval
                
            def __call__(self, state, data=None):
                # 使用 asyncio.get_event_loop().create_task() 而不是 run_until_complete()
                import asyncio
                
                # 获取当前事件循环
                loop = asyncio.get_event_loop()
                
                # 检查是否在事件循环内运行
                if loop.is_running():
                    # 如果事件循环已经在运行，使用 nest_asyncio 允许嵌套事件循环
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                    except ImportError:
                        # 如果 nest_asyncio 未安装，尝试使用替代方法
                        try:
                            # 尝试在已运行的事件循环中创建一个新的Future并等待其完成
                            future = asyncio.run_coroutine_threadsafe(
                                self.async_eval.evaluate_prompt(state, data), loop)
                            return future.result()
                        except Exception as e:
                            # 如果失败，返回默认值
                            print(f"Warning: Could not evaluate async prompt: {e}")
                            return 0.5  # 返回中等分数作为后备选项
                
                # 如果事件循环未运行或已经应用了nest_asyncio
                try:
                    return loop.run_until_complete(self.async_eval.evaluate_prompt(state, data))
                except RuntimeError as e:
                    if "This event loop is already running" in str(e):
                        # 如果仍然失败，尝试使用替代方法（最后的后备方案）
                        print("Warning: Using fallback evaluation method")
                        # 在这种极端情况下，提供一个基于规则的评估
                        return 0.5  # 返回中等分数
                    raise  # 重新抛出其他运行时错误
        
        return SyncEvaluator(self.async_evaluator)
    
    async def test_ollama_connection(self):
        """测试与Ollama服务的连接"""
        print("\n正在测试与Ollama的连接...")
        try:
            # 发送一个简单的请求来测试连接
            response = await self.llm.generate("Hello, are you connected?")
            if response and "text" in response:
                print("✅ 成功连接到Ollama服务！")
                print(f"Ollama响应: {response['text'][:100]}...")
                return True
            else:
                print("❌ 连接到Ollama服务，但未收到预期响应")
                print(f"收到的响应: {response}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到Ollama服务: {str(e)}")
            print("请确保Ollama服务正在运行 (ollama serve)")
            return False
    
    async def run_optimization(self, 
                            prompt: str, 
                            data: Optional[str] = None,
                            task_type: str = "classification",
                            iterations: int = 50,
                            time_limit: float = 60.0):
        """
        Run the complete optimization process.
        
        Args:
            prompt: Initial prompt to optimize.
            data: Sample data for context (optional).
            task_type: Type of task.
            iterations: Maximum number of iterations.
            time_limit: Maximum time in seconds.
            
        Returns:
            Dictionary with optimization results.
        """
        print(f"\n{'='*60}")
        print(f"Starting optimization for {task_type} task")
        print(f"Initial prompt: {prompt[:50]}...")
        print(f"{'='*60}\n")
        
        # Initialize the state
        initial_state = PromptState(prompt)
        
        # Set up evaluator
        await self.setup_evaluator()
        sync_evaluator = self.create_sync_evaluator()
        # Create components
        transition = StateTransition()
        reward_fn = RewardFunction(
            task_performance_fn=sync_evaluator,
            structural_weight=0.3,
            efficiency_weight=0.1
        )
        
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
        print("Running MCTS optimization...")
        start_time = time.time()
        best_state, stats = mcts_engine.optimize(initial_state)
        elapsed = time.time() - start_time
        
        print(f"Optimization completed in {elapsed:.2f}s")
        print(f"Iterations: {stats['iterations']}")
        print(f"Tree size: {stats['tree_size']} nodes")
        print(f"Best reward: {stats['best_reward']:.4f}")
        
        # Process final output
        root_node = mcts_engine._root_node
        output_result = self.output_service.generate_output(
            root_node=root_node,
            original_data=data,
            selection_strategy="composite",
            verification_level="standard"
        )
        
        # Compare with original
        comparison = self.output_service.compare_with_original(
            original_prompt=prompt,
            optimized_prompt=best_state.text
        )
        
        # Generate visualizations (添加错误处理)
        try:
            tree_viz = self.visualizer.generate_tree_visualization(root_node)
            if tree_viz:
                with open(self.output_dir / "tree_visualization.png", "wb") as f:
                    f.write(base64.b64decode(tree_viz))
        except Exception as e:
            print(f"Warning: Could not generate tree visualization: {e}")
            tree_viz = None
        
        try:
            top_trajectories = self.output_service.prompt_selector.analyze_trajectories(root_node)
            trajectory_viz = self.visualizer.generate_trajectory_visualization(top_trajectories)
            if trajectory_viz:
                with open(self.output_dir / "trajectory_visualization.png", "wb") as f:
                    f.write(base64.b64decode(trajectory_viz))
        except Exception as e:
            print(f"Warning: Could not generate trajectory visualization: {e}")
            trajectory_viz = None
        
        try:
            reward_viz = self.visualizer.generate_reward_progression_visualization(best_state)
            if reward_viz:
                with open(self.output_dir / "reward_progression.png", "wb") as f:
                    f.write(base64.b64decode(reward_viz))
        except Exception as e:
            print(f"Warning: Could not generate reward progression visualization: {e}")
            reward_viz = None
        
        try:
            component_viz = self.visualizer.generate_component_comparison_visualization(
                comparison["original_components"],
                comparison["optimized_components"]
            )
            if component_viz:
                with open(self.output_dir / "component_comparison.png", "wb") as f:
                    f.write(base64.b64decode(component_viz))
        except Exception as e:
            print(f"Warning: Could not generate component comparison visualization: {e}")
            component_viz = None
            
        print("\nVisualizations saved to", self.output_dir)
        
        # Print optimization results
        print("\nInitial prompt:")
        print(prompt)
        print("\nOptimized prompt:")
        print(best_state.text)
        print("\nFinal output:")
        print(output_result["final_output"])
        
        # Evaluate the optimized prompt with LLM
        print("\nEvaluating optimized prompt with LLM...")
        eval_result = await self.evaluate_with_llm(
            prompt=output_result["final_output"],
            original_prompt=prompt
        )
        
        # Compile comprehensive results
        result = {
            "initial_prompt": prompt,
            "optimized_prompt": best_state.text,
            "final_output": output_result["final_output"],
            "data": data,
            "task_type": task_type,
            "optimization_stats": stats,
            "processing_stats": output_result["processing_stats"],
            "comparison": comparison,
            "evaluation": eval_result,
            "visualizations": {
                "tree": tree_viz,
                "trajectories": trajectory_viz,
                "reward_progression": reward_viz,
                "component_comparison": component_viz
            }
        }
        
        return result
    
    async def evaluate_with_llm(self, prompt: str, original_prompt: str) -> Dict[str, Any]:
        """
        Evaluate a prompt using the LLM.
        
        Args:
            prompt: Prompt to evaluate.
            original_prompt: Original prompt for comparison.
            
        Returns:
            Evaluation results.
        """
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
            response = await self.llm.generate(evaluation_prompt)
            
            # Extract the numerical rating if present
            text = response.get("text", "")
            rating = None
            
            # Try to find a numerical rating
            import re
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
    
    async def check_gpu_status(self):
        """检查GPU状态和使用情况"""
        print("\n正在检查GPU状态...")
        
        try:
            # 尝试导入GPU相关库
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                print(f"✅ 检测到 {gpu_count} 个可用GPU:")
                print(f"   当前使用的设备: {device_name}")
                print(f"   CUDA 版本: {torch.version.cuda}")
                
                # 检查GPU内存使用情况
                try:
                    # 尝试获取GPU内存信息
                    memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # 转换为GB
                    memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3    # 转换为GB
                    print(f"   已分配内存: {memory_allocated:.2f} GB")
                    print(f"   已保留内存: {memory_reserved:.2f} GB")
                except:
                    print("   无法获取GPU内存信息")
                
                return True
            else:
                print("❌ 未检测到可用的GPU")
                print("   PyTorch版本: ", torch.__version__)
                return False
        except ImportError:
            print("❌ 未安装PyTorch或CUDA库")
            print("   请安装支持CUDA的PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return False
        except Exception as e:
            print(f"❌ 检查GPU状态时出错: {str(e)}")
            return False
    
    async def interactive_demo(self):
        """Run the simplified interactive demo."""
        print("\n" + "="*60)
        print("MCTS-Evo-Prompt Interactive Demo")
        print("="*60)
        
        # 先测试Ollama连接和GPU状态
        connected = await self.test_ollama_connection()
        if not connected:
            return
        
        await self.check_gpu_status()
        
        # 简化任务类型选择
        print("\n选择任务类型 (或按Enter使用自定义):")
        print("1. 分类 (Classification)")
        print("2. 情感分析 (Sentiment Analysis)")
        print("3. 摘要 (Summarization)")
        print("4. 信息提取 (Information Extraction)")
        print("5. 内容生成 (Content Generation)")
        print("6. 自定义任务 (Custom Task)")
        
        task_choice = input("输入选择 (默认: 6): ").strip() or "6"
        
        task_types = {
            "1": "classification",
            "2": "sentiment_analysis",
            "3": "summarization",
            "4": "extraction",
            "5": "generation",
            "6": "custom"
        }
        
        task_type = task_types.get(task_choice, "custom")
        print(f"\n已选择任务类型: {task_type}")
        
        # 获取提示词，提供默认选项
        print("\n输入您的提示词 (或按Enter使用默认提示词):")
        
        default_prompts = {
            "classification": "将以下文本分类为积极、消极或中性。",
            "sentiment_analysis": "分析以下文本的情感，判断其为积极、消极或中性。",
            "summarization": "简洁地总结以下文本的主要内容。",
            "extraction": "从以下文本中提取关键信息。",
            "generation": "根据以下指令生成内容。",
            "custom": "完成任务并提供清晰的回应。"
        }
        
        # 获取提示词输入
        print("输入提示词，输入空行结束:")
        
        prompt_lines = []
        while True:
            try:
                line = input()
                if not line.strip():
                    break
                prompt_lines.append(line)
            except EOFError:
                break
        
        # 使用默认提示词或用户输入
        if not prompt_lines:
            prompt = default_prompts[task_type]
            print(f"使用{task_type}的默认提示词:")
            print(prompt)
        else:
            prompt = "\n".join(prompt_lines)
        
        # 简化数据输入
        print("\n是否提供样本数据? (y/n, 默认: n):")
        data_choice = input().lower().strip() or "n"
        
        data = None
        if data_choice.startswith('y'):
            print("\n输入您的样本数据 (输入空行结束):")
            
            data_lines = []
            while True:
                try:
                    line = input()
                    if not line.strip():
                        break
                    data_lines.append(line)
                except EOFError:
                    break
            
            if data_lines:
                data = "\n".join(data_lines)
                print("\n您的样本数据:")
                print("-" * 40)
                print(data)
                print("-" * 40)
        
        # 简化参数设置，使用默认值
        print("\n优化参数 (按Enter使用默认值):")
        
        iterations = 30  # 默认值
        try:
            iterations_input = input(f"迭代次数 (默认: {iterations}): ").strip()
            if iterations_input:
                iterations = int(iterations_input)
        except ValueError:
            print(f"无效输入，使用默认值: {iterations}")
        
        time_limit = 40.0  # 默认值
        try:
            time_limit_input = input(f"时间限制(秒) (默认: {time_limit}): ").strip()
            if time_limit_input:
                time_limit = float(time_limit_input)
        except ValueError:
            print(f"无效输入，使用默认值: {time_limit}")
        
        print("\n开始优化，使用以下设置:")
        print(f"- 任务类型: {task_type}")
        print(f"- 迭代次数: {iterations}")
        print(f"- 时间限制: {time_limit} 秒")
        
        try:
            # 运行优化
            result = await self.run_optimization(
                prompt=prompt,
                data=data,
                task_type=task_type,
                iterations=iterations,
                time_limit=time_limit
            )
            
            # 显示结果摘要
            print("\n" + "="*60)
            print("优化结果")
            print("="*60)
            
            print("\n原始提示词:")
            print(result["initial_prompt"])
            
            print("\n优化后的提示词:")
            print(result["optimized_prompt"])
            
            print("\n结构改进:")
            if "structural_improvements" in result["comparison"]:
                for improvement in result["comparison"]["structural_improvements"]:
                    print(f"- {improvement}")
            else:
                print("没有识别到特定的结构改进。")
            
            # 简化LLM评估输出
            print("\nLLM评估摘要:")
            if "evaluation" in result and "evaluation_text" in result["evaluation"]:
                eval_text = result["evaluation"]["evaluation_text"]
                print(eval_text)
            else:
                print("LLM评估不可用。")
            
            if result.get("evaluation", {}).get("rating"):
                print(f"\n评分: {result['evaluation']['rating']}/10")
            
            # 显示优化统计的简化版本
            print("\n优化统计:")
            stats = result.get("optimization_stats", {})
            print(f"- 迭代次数: {stats.get('iterations', 'N/A')}")
            print(f"- 用时: {stats.get('time', 0):.2f}秒")
            print(f"- 最佳奖励值: {stats.get('best_reward', 0):.4f}")
            
            if any(viz is not None for viz in result["visualizations"].values()):
                print("\n可视化结果已保存到:", self.output_dir)
            
            # 简化测试流程
            test_choice = input("\n测试优化后的提示词? (y/n, 默认: n): ").lower().strip() or "n"
            
            if test_choice.startswith('y'):
                await self.test_with_llm(result["final_output"], data)
        
        except Exception as e:
            import traceback
            print(f"\n优化过程中出错: {e}")
            print(traceback.format_exc())
    
    async def test_with_llm(self, prompt: str, data: Optional[str] = None):
        """
        Test a prompt with the LLM.
        
        Args:
            prompt: Prompt to test.
            data: Sample data to include (optional).
        """
        print("\n" + "="*60)
        print("LLM Test")
        print("="*60)
        
        full_prompt = prompt
        if data:
            full_prompt = f"{prompt}\n\n{data}"
        
        print("\nSending prompt to LLM...")
        try:
            response = await self.llm.generate(full_prompt)
            print("\nLLM Response:")
            print(response.get("text", ""))
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Run the MCTS-Evo-Prompt demo."""
    parser = argparse.ArgumentParser(description="MCTS-Evo-Prompt Demo")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="mistral", help="Model ID")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--test-only", action="store_true", 
                        help="Only test connection to LLM and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed error information")
    
    args = parser.parse_args()
    
    # 初始化 Demo
    demo = MCTSEvoPromptDemo(
        llm_provider=args.provider,
        model_id=args.model,
        use_gpu=not args.no_gpu
    )
    
    # 如果只是测试连接
    if args.test_only:
        asyncio.run(demo.test_ollama_connection())
        asyncio.run(demo.check_gpu_status())
        return
    
    # 安装需要的包
    try:
        import nest_asyncio
        nest_asyncio.apply()
        print("已启用嵌套事件循环支持")
    except ImportError:
        print("Warning: nest_asyncio 未安装，可能导致异步操作问题")
        print("建议使用 pip install nest_asyncio 安装")
    
    # 运行交互式演示
    try:
        asyncio.run(demo.interactive_demo())
    except Exception as e:
        print(f"\n程序运行时出错: {e}")
        if args.debug:
            import traceback
            print("\n详细错误信息:")
            traceback.print_exc()
        else:
            print("使用 --debug 参数运行程序获取详细错误信息")
            
if __name__ == "__main__":
    main()