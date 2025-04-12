"""
Task runner for MCTS-Evo-Prompt system.
"""
import os
import sys
import json
import time
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
# Also add the parent directory if we're in backend
if project_root.name == "backend":
    sys.path.append(str(project_root.parent))

def setup_environment():
    """Set up the environment for running tasks"""
    # Ensure data directories exist
    data_dirs = [
        project_root / "data" / "tasks",
        project_root / "app" / "data" / "tasks",
        project_root / "output"
    ]
    
    for data_dir in data_dirs:
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure data/__init__.py includes load_dataset
    data_module_path = project_root / "data" / "__init__.py"
    if not data_module_path.exists():
        if not (project_root / "data").exists():
            (project_root / "data").mkdir(parents=True, exist_ok=True)
        
        with open(data_module_path, "w") as f:
            f.write("""\"\"\"
Data module for MCTS-Evo-Prompt.
\"\"\"
# Import and expose the load_dataset function
try:
    from .load_dataset import load_dataset
except ImportError:
    # Fallback implementation if the module isn't available
    import os
    import json
    from pathlib import Path
    
    def load_dataset(dataset_name, subset=None):
        \"\"\"Simple fallback implementation for load_dataset\"\"\"
        print(f"Using fallback load_dataset for {dataset_name}")
        
        # Create a simple dataset structure
        return {
            "train": [],
            "test": []
        }
""")

    # Create load_dataset.py if it doesn't exist
    load_dataset_path = project_root / "data" / "load_dataset.py"
    if not load_dataset_path.exists():
        with open(load_dataset_path, "w") as f:
            f.write("""\"\"\"
Data loading utilities for MCTS-Evo-Prompt.
This module provides a local implementation of the load_dataset function
that task files use to load datasets from various sources.
\"\"\"
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

def load_dataset(dataset_name: str, subset: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"
    Load a dataset by name.
    
    This function provides an alternative to Hugging Face datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        subset: Optional subset name (specific to certain datasets)
        
    Returns:
        Dictionary containing the dataset
    \"\"\"
    print(f"Loading dataset: {dataset_name}" + (f", subset: {subset}" if subset else ""))
    
    # Try to load from local files first
    local_dataset = _try_load_local_dataset(dataset_name, subset)
    if local_dataset is not None:
        return local_dataset
    
    # If not found, create a synthetic dataset
    return _create_synthetic_dataset(dataset_name, subset)

def _try_load_local_dataset(dataset_name: str, subset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    \"\"\"Try to load dataset from local data directory.\"\"\"
    # Normalize dataset name
    dataset_name = dataset_name.lower().replace("-", "_")
    
    # Common data file locations
    data_paths = [
        f"data/tasks/{dataset_name}.json",
        f"data/{dataset_name}.json",
        f"backend/data/tasks/{dataset_name}.json",
        f"backend/data/{dataset_name}.json", 
        f"app/data/tasks/{dataset_name}.json"
    ]
    
    # Check if any of the paths exist and load the data
    for path in data_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Loaded dataset {dataset_name} from {path}")
                
                # Handle subset if provided
                if subset and isinstance(data, dict) and subset in data:
                    return data[subset]
                    
                return data
            except Exception as e:
                print(f"Error loading dataset from {path}: {e}")
    
    return None

def _create_synthetic_dataset(dataset_name: str, subset: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"Create a minimal synthetic dataset for testing.\"\"\"
    print(f"Creating synthetic dataset for {dataset_name}")
    
    # Task-specific synthetic data
    if dataset_name == "trec":
        dataset = {
            "train": [
                {"text": "What is the capital of France?", "coarse_label": 0},
                {"text": "Who invented the telephone?", "coarse_label": 3}
            ],
            "test": [
                {"text": "What is the tallest mountain?", "coarse_label": 0},
                {"text": "When was the Declaration of Independence signed?", "coarse_label": 5}
            ]
        }
    elif dataset_name == "super_glue" and subset == "cb":
        dataset = {
            "train": [
                {"premise": "The man is sleeping.", "hypothesis": "The man is awake.", "label": 1},
                {"premise": "The bird is flying.", "hypothesis": "The bird has wings.", "label": 0}
            ],
            "validation": [
                {"premise": "The cat is on the mat.", "hypothesis": "The mat is under the cat.", "label": 0},
                {"premise": "It's raining outside.", "hypothesis": "The ground is wet.", "label": 0}
            ]
        }
    elif dataset_name == "biosses":
        dataset = {
            "train": [
                {"sentence1": "Protein kinase C (PKC) is activated by diacylglycerol.", 
                 "sentence2": "PKC activation is carried out by diacylglycerol.", 
                 "score": 4.0},
                {"sentence1": "BNIP3 interacts with LC3 and this leads to mitophagy.", 
                 "sentence2": "BNIP3 causes internalisation of LC3 in the mitochondria.", 
                 "score": 3.0}
            ]
        }
    elif dataset_name == "ncbi_disease":
        dataset = {
            "train": [
                {"tokens": ["Mutation", "in", "the", "APC", "gene", "causes", "colorectal", "cancer", "."],
                 "ner_tags": [0, 0, 0, 1, 2, 0, 1, 2, 0]},
                {"tokens": ["The", "BRCA1", "gene", "is", "linked", "to", "breast", "cancer", "."],
                 "ner_tags": [0, 1, 2, 0, 0, 0, 1, 2, 0]}
            ],
            "validation": [
                {"tokens": ["Cystic", "fibrosis", "is", "caused", "by", "a", "mutation", "in", "the", "CFTR", "gene", "."],
                 "ner_tags": [1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0]}
            ],
            "test": [
                {"tokens": ["Huntington's", "disease", "is", "a", "neurodegenerative", "disorder", "."],
                 "ner_tags": [1, 2, 0, 0, 0, 0, 0]}
            ]
        }
    elif dataset_name == "SetFit/subj":
        dataset = {
            "train": [
                {"text": "This movie was fantastic and I loved every minute of it.", "label": 1, "label_text": "Subjective"},
                {"text": "The temperature today is 25 degrees Celsius.", "label": 0, "label_text": "Objective"}
            ],
            "test": [
                {"text": "I think the book was boring and too long.", "label": 1, "label_text": "Subjective"},
                {"text": "The Earth revolves around the Sun.", "label": 0, "label_text": "Objective"}
            ]
        }
    elif dataset_name == "bigbio/med_qa":
        dataset = {
            "train": [
                {"question": "Which of the following is a symptom of pneumonia?",
                 "options": [{"key": "A", "value": "Fever"}, {"key": "B", "value": "Hair loss"}, 
                             {"key": "C", "value": "Skin rash"}, {"key": "D", "value": "Joint pain"}],
                 "answer": "Fever", "answer_idx": 0},
                {"question": "What is the function of insulin?",
                 "options": [{"key": "A", "value": "Decrease blood glucose"}, {"key": "B", "value": "Increase blood pressure"}, 
                             {"key": "C", "value": "Decrease heart rate"}, {"key": "D", "value": "Increase body temperature"}],
                 "answer": "Decrease blood glucose", "answer_idx": 0}
            ],
            "test": [
                {"question": "Which organ is primarily responsible for filtering blood?",
                 "options": [{"key": "A", "value": "Kidney"}, {"key": "B", "value": "Liver"}, 
                             {"key": "C", "value": "Spleen"}, {"key": "D", "value": "Lung"}],
                 "answer": "Kidney", "answer_idx": 0}
            ]
        }
    else:
        # Create a minimal dataset structure for any other dataset
        dataset = {
            "train": [
                {"question": "Sample question 1", "answer": "Sample answer 1"},
                {"question": "Sample question 2", "answer": "Sample answer 2"}
            ],
            "test": [
                {"question": "Test question 1", "answer": "Test answer 1"},
                {"question": "Test question 2", "answer": "Test answer 2"}
            ]
        }
    
    # Create directory to save synthetic data for future use
    if dataset_name:
        try:
            data_dir = Path("data/tasks")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{dataset_name}.json"
            if subset:
                filename = f"{dataset_name}_{subset}.json"
            
            with open(data_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
                
            print(f"Saved synthetic dataset to {data_dir / filename}")
        except Exception as e:
            print(f"Error saving synthetic dataset: {e}")
    
    return dataset
""")

# Set up environment before importing modules
setup_environment()

from app.core.mdp.state import PromptState
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction
from app.core.mdp.action import Action, create_action

from app.core.mcts.engine import MCTSEngine
from app.core.optimization.prompt_selector import PromptSelector
from app.core.optimization.token_optimizer import TokenOptimizer
from app.core.optimization.output_processor import OutputProcessor

from app.core.evolution.mutation import PromptMutator
from app.core.evolution.crossover import PromptCrossover
from app.core.evolution.selection import EvolutionSelector

from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider

from app.knowledge.error.error_collector import ErrorCollector
from app.knowledge.error.error_analyzer import ErrorAnalyzer
from app.knowledge.error.feedback_generator import FeedbackGenerator

from app.knowledge.domain.domain_knowledge import DomainKnowledgeManager
from app.knowledge.integration.integrator import PromptKnowledgeIntegrator

# Import task module and task-specific action generators
from tasks import get_task
from app.actions.task_actions import get_task_action_generator

class TaskRunner:
    """Runner for executing tasks with MCTS-Evo-Prompt."""
    
    def __init__(
        self, 
        task_name: str,
        data_path: str = None,
        llm_model: str = "mistral",
        output_dir: str = None,
        use_gpu: bool = True,
        task_configs: dict = None,
        debug_mode: bool = False
    ):
        """
        Initialize the task runner.
        
        Args:
            task_name: Name of the task to run.
            data_path: Path to task data file.
            llm_model: Name of the LLM model to use.
            output_dir: Directory to save output.
            use_gpu: Whether to use GPU.
            task_configs: Task-specific configurations.
            debug_mode: Whether to enable detailed debug logging.
        """
        self.task_name = task_name
        self.task_configs = task_configs or {}
        self.debug_mode = debug_mode
        
        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path(f"output/{task_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging directory
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging if debug mode is enabled
        if debug_mode:
            import logging
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_dir / "debug.log"),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger("task_runner")
            self.logger.debug(f"Debug logging enabled for {task_name}")
        
        # Register Ollama provider
        LLMFactory.register_provider("ollama", OllamaProvider)
        
        # Configure GPU usage
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["OMP_NUM_THREADS"] = "4"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Create LLM with optimized parameters
        llm_params = {
            "n_gpu_layers": -1 if use_gpu else 0,
            "n_ctx": 4096,
            "f16_kv": True
        }
        
        # Initialize LLM
        print(f"Initializing LLM (model: {llm_model})...")
        self.llm = LLMFactory.create(
            "ollama", 
            model_id=llm_model,
            **llm_params
        )
        
        # Initialize knowledge components
        self.error_collector = ErrorCollector(llm=self.llm)
        self.error_analyzer = ErrorAnalyzer()
        self.feedback_generator = FeedbackGenerator()
        self.knowledge_manager = DomainKnowledgeManager()
        self.knowledge_integrator = PromptKnowledgeIntegrator()
        
        # Find data path if not specified
        if not data_path:
            # Auto-determine data path for known tasks
            task_lower = task_name.lower()
            
            # Check common data file locations
            potential_paths = [
                f"data/tasks/{task_lower}.json",
                f"app/data/tasks/{task_lower}.json", 
                f"backend/data/tasks/{task_lower}.json",
                f"data/{task_lower}.json",
                f"backend/data/{task_lower}.json",
            ]
            
            # Add special handling for BigBench tasks
            if task_lower in [
                "penguins_in_a_table", "causal_judgment", "epistemic", 
                "geometric_shapes", "object_counting", "temporal_sequences"
            ]:
                potential_paths.insert(0, f"app/data/tasks/{task_lower}.json")
            
            # Find the first existing path
            for path in potential_paths:
                if Path(path).exists():
                    data_path = path
                    print(f"Found data file: {data_path}")
                    break
            
            # If still not found, use a default path (task will handle missing file)
            if not data_path:
                data_path = f"data/tasks/{task_lower}.json"
                print(f"Using default data path: {data_path}")
        
        print(f"Initializing task: {task_name}")
        try:
            # Try to initialize the task
            self.task = get_task(task_name)(
                train_size=100,  # Use 100 examples for training
                eval_size=10,    # Use 10 examples for evaluation
                test_size=10,    # Use 10 examples for testing
                data_dir=data_path
            )
        except Exception as e:
            print(f"Error initializing task: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize task {task_name}. Please check that the task module exists and is correctly implemented.")
        
        # Initialize optimization components
        self.prompt_selector = PromptSelector()
        self.token_optimizer = TokenOptimizer()
        self.output_processor = OutputProcessor(token_optimizer=self.token_optimizer)
        
        # Cache for evaluations to avoid redundant API calls
        self.evaluation_cache = {}
        
        print(f"Task runner initialized for {task_name}")
    
    async def test_llm_connection(self):
        """Test connection to the LLM with a more robust check."""
        print("\nTesting connection to LLM...")
        try:
            # Request a specific response to verify the connection is truly working
            response = await self.llm.generate("Respond with 'CONNECTION_OK' if you can read this message.")
            
            if response and isinstance(response, dict) and "text" in response:
                # Check if the response contains meaningful text (not just an empty string)
                if len(response["text"].strip()) > 10:  # More than 10 chars means actual content
                    print("✅ Successfully connected to LLM!")
                    return True
                else:
                    print("❌ Connected to LLM, but received empty or very short response")
                    print(f"Response: {response}")
                    return False
            else:
                print("❌ Connected to LLM, but received unexpected response format")
                print(f"Response: {response}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to LLM: {str(e)}")
            print("Make sure the LLM service is running")
            return False
    
    async def check_gpu_status(self):
        """Check GPU status and availability."""
        print("\nChecking GPU status...")
        try:
            # Try to import GPU-related libraries
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                print(f"✅ Detected {gpu_count} available GPU(s):")
                print(f"   Current device: {device_name}")
                print(f"   CUDA version: {torch.version.cuda}")
                
                # Check GPU memory usage
                try:
                    memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # Convert to GB
                    memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3    # Convert to GB
                    print(f"   Allocated memory: {memory_allocated:.2f} GB")
                    print(f"   Reserved memory: {memory_reserved:.2f} GB")
                except:
                    print("   Unable to get GPU memory information")
                
                return True
            else:
                print("❌ No available GPUs detected")
                print("   PyTorch version: ", torch.__version__)
                return False
        except ImportError:
            print("❌ PyTorch or CUDA libraries not installed")
            print("   Please install PyTorch with CUDA support")
            return False
        except Exception as e:
            print(f"❌ Error checking GPU status: {str(e)}")
            return False
    
    async def evaluate_prompt(self, prompt: str, examples, collect_errors: bool = False):
        """
        Evaluate a prompt on the given examples.
        
        Args:
            prompt: The prompt to evaluate.
            examples: List of examples to evaluate on.
            collect_errors: Whether to collect errors for analysis.
            
        Returns:
            Performance metrics and optionally error information.
        """
        # Check if examples is empty or None
        if not examples:
            print("Warning: No examples provided for evaluation")
            return {"metrics": {"accuracy": 0}, "errors": []}
        
        responses = []
        labels = []
        questions = []
        errors = []
        
        print(f"Evaluating prompt on {len(examples)} examples...")
        
        # Get batch size from task configuration or default to 3
        batch_size = self.task_configs.get("batch_size", 3)
        
        # Process examples in batches to improve efficiency
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i+batch_size]
            batch_prompts = []
            batch_expected_answers = []
            batch_questions = []
            
            for example in batch_examples:
                if isinstance(example, dict) and 'question' in example and 'answer' in example:
                    full_prompt = f"{prompt}\n\n{example['question']}"
                    batch_prompts.append(full_prompt)
                    batch_expected_answers.append(example['answer'])
                    batch_questions.append(example.get('question', ''))
                else:
                    # Skip if example doesn't have the expected format
                    print(f"Warning: Example doesn't have the expected format: {example}")
            
            # Generate responses in parallel for the batch
            batch_responses = []
            for full_prompt in batch_prompts:
                try:
                    response = await self.llm.generate(full_prompt)
                    response_text = response.get("text", "")
                    batch_responses.append(response_text)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    batch_responses.append("ERROR")
            
            # Add batch results to overall results
            responses.extend(batch_responses)
            labels.extend(batch_expected_answers)
            questions.extend(batch_questions)
            
            # Collect errors if needed
            if collect_errors:
                for j, (question, expected, actual) in enumerate(zip(batch_questions, batch_expected_answers, batch_responses)):
                    try:
                        cleaned_actual = self.task.clean_response(actual) if hasattr(self.task, 'clean_response') else actual
                        cleaned_expected = self.task.clean_labels([expected])[0] if hasattr(self.task, 'clean_labels') else expected
                        
                        if cleaned_actual != cleaned_expected:
                            errors.append({
                                "id": f"ex_{i+j}",
                                "text": question,
                                "expected": cleaned_expected,
                                "actual": cleaned_actual,
                                "error_type": self._determine_error_type(cleaned_actual, cleaned_expected, question)
                            })
                    except Exception as e:
                        print(f"Error processing error collection: {e}")
                        # Continue with next example
        
        # Clean responses and labels
        try:
            cleaned_responses = self.task.batch_clean_responses(responses) if hasattr(self.task, 'batch_clean_responses') else responses
        except Exception as e:
            print(f"Error cleaning responses: {e}")
            cleaned_responses = responses
            
        try:
            cleaned_labels = self.task.clean_labels(labels) if hasattr(self.task, 'clean_labels') else labels
        except Exception as e:
            print(f"Error cleaning labels: {e}")
            cleaned_labels = labels
        
        # Calculate metrics
        metrics = {}
        try:
            # First try the task-specific calculation if it has questions
            if hasattr(self.task, 'cal_metric') and questions:
                metrics_result = self.task.cal_metric(cleaned_responses, cleaned_labels, questions)
                if isinstance(metrics_result, tuple):
                    metrics["f1"] = metrics_result[0]
                    metrics["precision"] = metrics_result[1] if len(metrics_result) > 1 else None
                    metrics["recall"] = metrics_result[2] if len(metrics_result) > 2 else None
                else:
                    metrics["accuracy"] = metrics_result
            else:
                # Fallback to standard accuracy calculation
                correct = sum(1 for p, l in zip(cleaned_responses, cleaned_labels) if p == l)
                metrics["accuracy"] = correct / len(cleaned_labels) if cleaned_labels else 0
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics["error"] = str(e)
            metrics["accuracy"] = 0
        
        result = {"metrics": metrics}
        
        # Add error information if requested
        if collect_errors and errors:
            result["errors"] = errors
            result["error_count"] = len(errors)
        
        return result
    
    def _determine_error_type(self, actual, expected, question):
        """Determine the type of error based on the question and responses."""
        # For tabular data questions
        if "table" in question.lower() or "penguin" in question.lower():
            if "count" in question.lower() or "how many" in question.lower():
                return "counting_error"
            elif "highest" in question.lower() or "lowest" in question.lower() or "most" in question.lower():
                return "comparison_error"
            elif "average" in question.lower() or "mean" in question.lower():
                return "calculation_error"
            else:
                return "lookup_error"
        
        # For other question types
        elif "yes" in str(expected).lower() or "no" in str(expected).lower():
            return "classification_error"
        elif any(c.isdigit() for c in str(expected)):
            return "numerical_error"
        else:
            return "factual_error"
    
    def build_performance_evaluator(self, collect_errors=True):
        """
        Build a performance evaluator function for the MCTS.
        
        Args:
            collect_errors: Whether to collect errors for error feedback.
            
        Returns:
            A function that evaluates a state and returns a score.
        """
        # Get evaluation examples - use eval_set first, fallback to eval_size
        eval_examples = []
        
        # Try different ways to get evaluation examples
        if hasattr(self.task, 'eval_set'):
            eval_examples = self.task.eval_set
        elif hasattr(self.task, 'dataset') and isinstance(self.task.dataset, dict) and 'eval' in self.task.dataset:
            eval_examples = self.task.dataset['eval']
        elif hasattr(self.task, 'eval_size'):
            eval_examples = self.task.eval_size
        
        # If we still don't have examples, create some minimal ones for testing
        if not eval_examples:
            print("Warning: No evaluation examples found, creating minimal examples")
            eval_examples = [
                {"question": "Sample question 1", "answer": "Sample answer 1"},
                {"question": "Sample question 2", "answer": "Sample answer 2"}
            ]
        
        # Limit to a reasonable number for efficiency
        eval_examples = eval_examples[:5]
        
        # Initialize error collection buffer
        error_buffer = []
        
        async def _evaluate_async(state_text):
            """Internal async evaluation function."""
            result = await self.evaluate_prompt(state_text, eval_examples, collect_errors=collect_errors)
            
            # Store errors if available
            if collect_errors and "errors" in result:
                error_buffer.clear()  # Clear previous errors
                error_buffer.extend(result["errors"])
            
            # Return the main metric
            metrics = result.get("metrics", {})
            score = metrics.get("accuracy", 0) or metrics.get("f1", 0)
            return score
        
        def evaluator(state: PromptState, data=None):
            """Evaluate a prompt state using the task's evaluation set."""
            # Check cache first
            if state.state_id in self.evaluation_cache:
                return self.evaluation_cache[state.state_id]
                
            # Use asyncio to run async function in sync context
            import asyncio
            
            # Check if we're in an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an event loop, so we need to use run_until_complete carefully
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                    except ImportError:
                        print("Warning: nest_asyncio module not found. Using new event loop.")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Evaluate prompt
            try:
                score = loop.run_until_complete(_evaluate_async(state.text))
            except Exception as e:
                print(f"Error evaluating prompt: {e}")
                score = 0
            
            # Cache the result
            self.evaluation_cache[state.state_id] = score
            
            # Log evaluation
            if self.debug_mode:
                self.logger.debug(f"Evaluated state {state.state_id[:8]} - Score: {score:.4f}")
                if error_buffer:
                    self.logger.debug(f"Found {len(error_buffer)} errors during evaluation")
            
            return score
        
        # Add getter for error buffer
        evaluator.get_errors = lambda: list(error_buffer)
        
        return evaluator
    
    def get_task_specific_knowledge(self):
        """
        Get task-specific domain knowledge, with fallbacks.
        
        Returns:
            List of domain knowledge items for the current task.
        """
        # Try to load from knowledge base file
        domain_knowledge = []
        try:
            # Check multiple potential locations
            kb_paths = [
                Path("app/data/knowledge_base/domain_knowledge/task_prompts.json"),
                Path("data/knowledge_base/domain_knowledge/task_prompts.json"),
                Path("backend/app/data/knowledge_base/domain_knowledge/task_prompts.json"),
                Path("backend/data/knowledge_base/domain_knowledge/task_prompts.json")
            ]
            
            kb_file = None
            for path in kb_paths:
                if path.exists():
                    kb_file = path
                    break
                    
            if kb_file:
                with open(kb_file, 'r') as f:
                    kb_data = json.load(f)
                    if self.task_name.lower() in kb_data:
                        domain_knowledge = kb_data[self.task_name.lower()].get("domain_knowledge", [])
                        if domain_knowledge:
                            print(f"✅ Loaded {len(domain_knowledge)} domain knowledge items from knowledge base")
            
            # Check if we got any knowledge
            if not domain_knowledge:
                # Try task-specific knowledge directory paths
                kb_dir_paths = [
                    Path(f"app/data/knowledge_base/tasks/{self.task_name.lower()}"),
                    Path(f"data/knowledge_base/tasks/{self.task_name.lower()}"),
                    Path(f"backend/app/data/knowledge_base/tasks/{self.task_name.lower()}"),
                    Path(f"backend/data/knowledge_base/tasks/{self.task_name.lower}")
                ]
                
                for task_kb_dir in kb_dir_paths:
                    if task_kb_dir.exists() and task_kb_dir.is_dir():
                        for kb_file in task_kb_dir.glob("*.json"):
                            try:
                                with open(kb_file, 'r') as f:
                                    kb_items = json.load(f)
                                    if isinstance(kb_items, list):
                                        domain_knowledge.extend(kb_items)
                                    elif isinstance(kb_items, dict) and "knowledge" in kb_items:
                                        domain_knowledge.extend(kb_items["knowledge"])
                            except Exception as e:
                                print(f"Error loading knowledge from {kb_file}: {e}")
                        
                        if domain_knowledge:
                            print(f"✅ Loaded {len(domain_knowledge)} domain knowledge items from task directory")
                            break
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
        
        # If still no knowledge, use hardcoded fallbacks
        if not domain_knowledge:
            if self.task_name.lower() == "penguins_in_a_table":
                domain_knowledge = [
                    {
                        "type": "conceptual_knowledge",
                        "statement": "Tables contain structured data with rows and columns. Column headers describe the data in each column.",
                        "entities": ["table", "column", "row", "header"]
                    },
                    {
                        "type": "procedural_knowledge",
                        "statement": "Process table data by first identifying column headers and data types, then extracting relevant cells based on the query.",
                        "procedure_topic": "table_processing",
                        "procedure_steps": [
                            "Identify column headers and what they represent",
                            "Determine data types for each column (numeric, categorical, etc.)",
                            "Extract relevant cells based on the query",
                            "Apply operations (counting, comparison, calculation) as needed",
                            "Verify your answer against the table data"
                        ]
                    },
                    {
                        "type": "entity_classification",
                        "statement": "Table queries can be categorized as lookup (direct cell retrieval), counting (how many rows/items match conditions), comparison (which row has max/min values), or calculation (average, sum, etc.).",
                        "entities": ["lookup query", "counting query", "comparison query", "calculation query"]
                    },
                    {
                        "type": "format_specification",
                        "statement": "For table data questions, first state your understanding of the table structure, then explain your approach to answering the question, and finally provide the exact answer with any supporting values from the table.",
                        "format_rules": [
                            "Start with a clear understanding of the table columns and data",
                            "Explain your approach to finding the answer",
                            "Provide the exact answer with supporting evidence"
                        ]
                    }
                ]
                print("ℹ️ Using fallback domain knowledge for penguins_in_a_table task")
            
            elif "count" in self.task_name.lower() or "counting" in self.task_name.lower():
                domain_knowledge = [
                    {
                        "type": "procedural_knowledge",
                        "statement": "When counting objects, identify each distinct item to be counted, track them systematically, and verify the count.",
                        "procedure_topic": "counting",
                        "procedure_steps": [
                            "Identify all items that match the counting criteria",
                            "Count systematically to avoid missing items or counting twice",
                            "Group similar items if helpful",
                            "Verify your count by rechecking"
                        ]
                    }
                ]
                print(f"ℹ️ Using fallback domain knowledge for {self.task_name} task")
            elif self.task_name.lower() == "ncbi":
                domain_knowledge = [
                    {
                        "type": "task_instruction",
                        "statement": "For named entity recognition in biomedical text, identify all disease entities mentioned in the text.",
                        "format_rules": [
                            "Return all disease entities in a comma-separated list within curly braces",
                            "Normalize entity spelling and format",
                            "If no disease entities are present, return an empty list: {}"
                        ]
                    }
                ]
                print(f"ℹ️ Using fallback domain knowledge for ncbi task")
            elif self.task_name.lower() == "trec":
                domain_knowledge = [
                    {
                        "type": "task_instruction",
                        "statement": "The TREC question classification task requires determining what type of information a question is asking for.",
                        "categories": [
                            "A: Abbreviation - seeking expansion of acronyms or abbreviations",
                            "B: Entity - questions about entities like products, organizations",
                            "C: Description - seeking definitions, explanations, or conceptual information",
                            "D: Human - questions about specific people or categories of people",
                            "E: Location - geographical questions about places, locations, etc.",
                            "F: Numeric - questions seeking numerical answers like dates, amounts, etc."
                        ]
                    }
                ]
                print(f"ℹ️ Using fallback domain knowledge for trec task")
        
        return domain_knowledge
    
    def _create_reward_booster(self, task_name):
        """Create a reward booster function specific to the task."""
        
        def boost_penguins_reward(state, base_reward):
            """Boost reward for penguins_in_a_table task based on prompt quality."""
            boost = 0.0
            
            # Check for table-specific content
            if "column" in state.text.lower() and "row" in state.text.lower():
                boost += 0.1
                
            # Check for data analysis terminology
            data_terms = ["data", "table", "analyze", "calculate", "compare"]
            term_count = sum(1 for term in data_terms if term in state.text.lower())
            boost += 0.02 * term_count
                
            # Check for step-by-step instruction
            if "step-by-step" in state.text.lower() or "step by step" in state.text.lower():
                boost += 0.1
                
            return min(0.3, boost)  # Cap at 0.3
        
        def boost_counting_reward(state, base_reward):
            """Boost reward for object_counting task based on prompt quality."""
            boost = 0.0
            
            # Check for counting methodology
            count_terms = ["count", "enumerate", "tally", "total", "sum"]
            term_count = sum(1 for term in count_terms if term in state.text.lower())
            boost += 0.02 * term_count
            
            # Check for grouping strategy mention
            if "group" in state.text.lower() or "categor" in state.text.lower():
                boost += 0.1
                
            # Check for verification steps
            if "verify" in state.text.lower() or "recheck" in state.text.lower():
                boost += 0.1
                
            return min(0.3, boost)  # Cap at 0.3
        
        def boost_subj_reward(state, base_reward):
            """Boost reward for subjective classification task based on prompt quality."""
            boost = 0.0
            
            # Check for classification terminology
            class_terms = ["subjective", "objective", "opinion", "fact"]
            term_count = sum(1 for term in class_terms if term in state.text.lower())
            boost += 0.03 * term_count
            
            # Check for analysis instructions
            if "analyze" in state.text.lower() and "text" in state.text.lower():
                boost += 0.1
                
            # Check for classification criteria
            if "criteria" in state.text.lower() or "indicator" in state.text.lower():
                boost += 0.1
                
            return min(0.3, boost)  # Cap at 0.3
        
        def boost_ncbi_reward(state, base_reward):
            """Boost reward for NCBI disease entity recognition task."""
            boost = 0.0
            
            # Check for entity recognition terminology
            ner_terms = ["entity", "entities", "disease", "recognition", "extract", "identify"]
            term_count = sum(1 for term in ner_terms if term in state.text.lower())
            boost += 0.02 * term_count
            
            # Check for formatting instructions
            if "{" in state.text and "}" in state.text:
                boost += 0.1
                
            # Check for normalization instructions
            if "normalize" in state.text.lower() or "standardize" in state.text.lower():
                boost += 0.1
                
            return min(0.3, boost)  # Cap at 0.3
        
        def boost_cb_reward(state, base_reward):
            """Boost reward for CB textual entailment task."""
            boost = 0.0
            
            # Check for entailment terminology
            entail_terms = ["entailment", "contradiction", "neutral", "hypothesis", "premise"]
            term_count = sum(1 for term in entail_terms if term in state.text.lower())
            boost += 0.03 * term_count
            
            # Check for reasoning instructions
            if "reason" in state.text.lower() or "relationship" in state.text.lower():
                boost += 0.1
                
            # Check for step-by-step analysis mention
            if "step" in state.text.lower() and "analysis" in state.text.lower():
                boost += 0.1
                
            return min(0.3, boost)  # Cap at 0.3
        
        def boost_trec_reward(state, base_reward):
            """Boost reward for TREC question classification task."""
            boost = 0.0
            
            # Check for question classification terminology
            class_terms = ["classify", "category", "question type", "question classification"]
            term_count = sum(1 for term in class_terms if term in state.text.lower())
            boost += 0.03 * term_count
            
            # Check for category mentions
            categories = ["abbreviation", "entity", "description", "human", "location", "numeric"]
            cat_count = sum(1 for cat in categories if cat in state.text.lower())
            boost += 0.02 * cat_count
                
            return min(0.3, boost)  # Cap at 0.3
        
        # Map task names to booster functions
        boosters = {
            "penguins_in_a_table": boost_penguins_reward,
            "object_counting": boost_counting_reward,
            "subj": boost_subj_reward,
            "ncbi": boost_ncbi_reward,
            "cb": boost_cb_reward,
            "trec": boost_trec_reward
        }
        
        # Return appropriate booster or default no-boost function
        return boosters.get(task_name.lower(), lambda state, reward: 0.0)
    
    def _get_task_specific_params(self):
        """Get task-specific parameters."""
        # Default parameters - with higher values for more extensive search
        default_params = {
            "max_depth": 6,           # Deeper search
            "max_children": 10,       # More branching factor
            "exploration_weight": 1.5, # Higher exploration
            "knowledge_rate": 0.6,    # More knowledge integration
            "batch_size": 5,
            "iterations": 50,         # More iterations
            "time_limit": 600         # Longer time limit
        }
        
        # Task-specific overrides with more aggressive parameters
        task_params = {
            "penguins_in_a_table": {
                "max_depth": 6,
                "max_children": 12,
                "exploration_weight": 1.4,
                "knowledge_rate": 0.7,
                "time_limit": 600,
                "iterations": 60
            },
            "object_counting": {
                "max_depth": 5,
                "max_children": 10,
                "exploration_weight": 1.5,
                "knowledge_rate": 0.6,
                "time_limit": 500,
                "iterations": 50
            },
            "subj": {
                "max_depth": 5,
                "max_children": 10,
                "exploration_weight": 1.5,
                "knowledge_rate": 0.6,
                "time_limit": 600,
                "iterations": 50
            },
            # Add other tasks with optimized parameters
            "trec": {
                "max_depth": 5,
                "max_children": 8,
                "exploration_weight": 1.5,
                "knowledge_rate": 0.6,
                "time_limit": 500,
                "iterations": 45
            },
            "ncbi": {
                "max_depth": 5,
                "max_children": 8,
                "exploration_weight": 1.4,
                "knowledge_rate": 0.7,
                "time_limit": 500,
                "iterations": 45
            },
            "cb": {
                "max_depth": 5,
                "max_children": 8,
                "exploration_weight": 1.4,
                "knowledge_rate": 0.6,
                "time_limit": 500,
                "iterations": 45
            }
        }
        
        # Get parameters for current task, fallback to defaults
        return task_params.get(self.task_name.lower(), default_params)
    
    async def run_optimization(self, iterations=30, time_limit=300):
        """
        Run the MCTS optimization process.
        
        Args:
            iterations: Maximum number of iterations.
            time_limit: Maximum time in seconds.
            
        Returns:
            Optimization results.
        """
        print(f"\n{'='*60}")
        print(f"Starting optimization for {self.task_name}")
        print(f"{'='*60}\n")
        
        # 1. INPUT PROCESSING AND INITIALIZATION
        # --------------------------------------
        print("[Phase 1] Input Processing and Initialization")

        # 1.1 Load task-specific domain knowledge
        domain_knowledge = self.get_task_specific_knowledge()
        
        # Get initial prompt
        initial_prompt = None
        try:
            # Check multiple potential locations
            kb_paths = [
                Path("app/data/knowledge_base/domain_knowledge/task_prompts.json"),
                Path("data/knowledge_base/domain_knowledge/task_prompts.json"),
                Path("backend/app/data/knowledge_base/domain_knowledge/task_prompts.json"),
                Path("backend/data/knowledge_base/domain_knowledge/task_prompts.json")
            ]
            
            kb_file = None
            for path in kb_paths:
                if path.exists():
                    kb_file = path
                    break
                    
            if kb_file:
                with open(kb_file, 'r') as f:
                    kb_data = json.load(f)
                    if self.task_name.lower() in kb_data:
                        initial_prompt = kb_data[self.task_name.lower()].get("initial_prompt")
                        if initial_prompt:
                            print(f"✅ Loaded task-specific initial prompt")
        except Exception as e:
            print(f"Error loading initial prompt: {e}")
        
        # Fallback to default prompt if no task-specific prompt is found
        if not initial_prompt:
            if hasattr(self.task, 'get_initial_prompt'):
                initial_prompt = self.task.get_initial_prompt()
            else:
                # Task-specific default prompts
                task_lower = self.task_name.lower()
                if task_lower == "penguins_in_a_table":
                    initial_prompt = "You are tasked with analyzing tabular data about penguins. First identify the structure of the table, then carefully extract the relevant information to answer the question. Provide a step-by-step analysis and verify your answer against the data in the table."
                elif task_lower == "ncbi":
                    initial_prompt = "Identify all disease entities mentioned in the text. Return your answer as a list of entities inside curly braces, like this: {entity1, entity2}. If no disease entities are present, return an empty list: {}"
                elif task_lower == "trec":
                    initial_prompt = "Classify the following question into one of these categories: (A) Abbreviation, (B) Entity, (C) Description and abstract concept, (D) Human being, (E) Location, or (F) Numeric value."
                elif task_lower == "object_counting":
                    initial_prompt = "Count all the instances of the specified objects in the description. Provide a clear step-by-step counting process to ensure accuracy."
                else:
                    initial_prompt = f"Answer the following {self.task_name} question accurately and concisely."
            
            print(f"Using default prompt for {self.task_name}")
        
        print(f"Initial prompt: {initial_prompt}")
        
        # 1.2 Apply pre-trained model enhancement
        print("\n[Phase 1.3] Applying Initial Prompt Expansion (Pre-trained Model)")
        try:
            # Try to import required modules, with fallbacks
            try:
                from app.core.input.model_trainer import PromptModelTrainer
                from app.core.input.task_analyzer import TaskAnalyzer
                from app.core.input.prompt_expander import PromptExpander
                
                # First analyze the task
                analyzer = TaskAnalyzer()
                task_analysis = analyzer.analyze(initial_prompt)
                print(f"Task analysis: Type={task_analysis['task_type']}, Confidence={task_analysis['task_confidence']:.2f}")
                
                # Attempt neural model expansion first
                try:
                    trainer = PromptModelTrainer()
                    enhanced_prompt = trainer.expand_prompt(initial_prompt)
                    
                    if enhanced_prompt and len(enhanced_prompt) > len(initial_prompt):
                        print("✅ Successfully enhanced with neural model")
                        initial_prompt = enhanced_prompt
                    else:
                        # Fallback to rule-based expansion
                        expander = PromptExpander()
                        enhanced_prompt = expander.expand(initial_prompt, task_analysis)
                        
                        if enhanced_prompt and len(enhanced_prompt) > len(initial_prompt):
                            print("✅ Successfully enhanced with rule-based expander")
                            initial_prompt = enhanced_prompt
                        else:
                            print("⚠️ Prompt expansion had no effect, using original prompt")
                except Exception as inner_e:
                    print(f"Neural model unavailable: {inner_e}")
                    
                    # Fallback to rule-based expansion
                    expander = PromptExpander()
                    enhanced_prompt = expander.expand(initial_prompt, task_analysis)
                    
                    if enhanced_prompt and len(enhanced_prompt) > len(initial_prompt):
                        print("✅ Successfully enhanced with rule-based expander")
                        initial_prompt = enhanced_prompt
                    else:
                        print("⚠️ Prompt expansion had no effect, using original prompt")
            except ImportError:
                print("⚠️ Prompt expansion modules not available, enhancing prompt manually")
                
                # Manual prompt enhancement based on task type
                task_lower = self.task_name.lower()
                if task_lower == "penguins_in_a_table":
                    if "step-by-step" not in initial_prompt.lower():
                        initial_prompt += "\n\nFollow a step-by-step approach: 1) Identify the column headers and data types, 2) Extract relevant information based on the question, 3) Apply the appropriate operations (counting, filtering, etc.), 4) Verify your answer against the table data."
                elif task_lower == "ncbi":
                    if "normalize" not in initial_prompt.lower():
                        initial_prompt += "\n\nMake sure to normalize disease entity names and remove any duplicates. Return the complete list of unique disease mentions."
                
                print("✅ Enhanced prompt manually")
        except Exception as e:
            print(f"⚠️ Prompt expansion not available: {e}")
        
        print("\nEnhanced initial prompt:")
        print("-" * 40)
        print(initial_prompt)
        print("-" * 40)
        
        # 2. MDP FRAMEWORK CONSTRUCTION
        # -----------------------------
        print("\n[Phase 2] MDP Framework Construction")
        
        # Initialize state
        initial_state = PromptState(initial_prompt)
        
        # Setup transition function
        transition = StateTransition()
        
        # Create performance evaluator with error collection
        performance_evaluator = self.build_performance_evaluator(collect_errors=True)
        
        # 3. MCTS STRATEGIC PLANNING WITH EVOLUTIONARY ALGORITHMS
        # -------------------------------------------------------
        print("\n[Phase 3] MCTS Strategic Planning with Evolutionary Algorithms")
        
        # Set up task-specific parameters
        task_specific_params = self._get_task_specific_params()
        
        # Initialize deep exploration parameters
        max_depth = task_specific_params.get("max_depth", 4)
        max_children = task_specific_params.get("max_children", 5)
        exploration_weight = task_specific_params.get("exploration_weight", 1.41)
        
        # Create a knowledge integration strategy based on task type
        if self.task_name.lower() == "penguins_in_a_table":
            print("\n[Phase 1.1.1] Setting up table analysis knowledge integration strategy")
            # Add specific knowledge integration for table tasks
            knowledge_integration_strategy = "early"  # Integrate knowledge early in optimization
        elif self.task_name.lower() == "object_counting":
            print("\n[Phase 1.1.2] Setting up counting knowledge integration strategy")
            knowledge_integration_strategy = "error_guided"  # Integrate knowledge based on errors
        elif self.task_name.lower() == "subj":
            print("\n[Phase 1.1.3] Setting up text classification knowledge integration strategy")
            knowledge_integration_strategy = "interleaved"  # Integrate throughout optimization
        elif self.task_name.lower() == "ncbi":
            print("\n[Phase 1.1.4] Setting up biomedical NER knowledge integration strategy")
            knowledge_integration_strategy = "early"  # Early knowledge integration for NER
        elif self.task_name.lower() == "cb":
            print("\n[Phase 1.1.5] Setting up textual entailment knowledge integration strategy")
            knowledge_integration_strategy = "interleaved"  # Integrate throughout for reasoning
        else:
            knowledge_integration_strategy = "adaptive"  # Default adaptive strategy

        print(f"Using knowledge integration strategy: {knowledge_integration_strategy}")

        # Create reward function with task-specific booster
        reward_booster = self._create_reward_booster(self.task_name)
        reward_fn = RewardFunction(
            task_performance_fn=performance_evaluator,
            structural_weight=0.25,  # Reduced weight
            efficiency_weight=0.05,  # Reduced weight
            reward_booster=reward_booster  # Add booster
        )

        # Set up evolution configuration with more task-specific settings
        evolution_config = {
            "adaptive_adjustment": True,  # Enable dynamic adjustment
            "mutation_rate": 0.25,        # Initial mutation rate
            "crossover_rate": 0.25,       # Initial crossover rate
            "error_feedback_rate": 0.5,   # Higher initial error feedback rate
            "domain_knowledge": domain_knowledge,
            "knowledge_integration_rate": task_specific_params.get("knowledge_rate", 0.6),
            "knowledge_integration_strategy": knowledge_integration_strategy
        }
        
        # Get task-specific action generator
        try:
            task_action_generator = get_task_action_generator(self.task_name.lower())
        except Exception as e:
            print(f"Error getting task action generator: {e}")
            print("Using default action generator")
            task_action_generator = None
        
        # Create error feedback components
        error_collector = self.error_collector
        error_analyzer = self.error_analyzer
        feedback_generator = self.feedback_generator
        
        # Define error feedback function
        def error_feedback_fn(state: PromptState):
            """Generate error feedback based on evaluation."""
            # Get errors from the evaluator
            errors = performance_evaluator.get_errors()
            
            if not errors:
                if self.debug_mode:
                    self.logger.debug("No errors found for error feedback")
                return []
            
            if self.debug_mode:
                self.logger.debug(f"Processing {len(errors)} errors for feedback")
            
            # Analyze errors
            try:
                error_analysis = error_analyzer.analyze_errors(errors)
                
                # Generate feedback
                feedback_items = feedback_generator.generate_feedback(error_analysis)
                
                # Convert feedback to actions
                actions = feedback_generator.map_feedback_to_actions(feedback_items)
                
                if self.debug_mode:
                    self.logger.debug(f"Generated {len(actions)} feedback actions")
                
                return actions
            except Exception as e:
                print(f"Error in feedback generation: {e}")
                return []
        
        # Create MCTS engine with all components
        mcts_engine = MCTSEngine(
            transition=transition,
            reward_function=reward_fn,
            max_iterations=iterations,
            time_limit=time_limit,
            exploration_weight=exploration_weight,
            max_depth=max_depth,  # Explicitly pass max_depth
            max_children_per_expansion=max_children,
            evolution_config=evolution_config,
            action_generator=task_action_generator,
            error_feedback_fn=error_feedback_fn  # Add error feedback function
        )
        
        # Run optimization
        print("\n[Phase 3.1] Running MCTS Core Algorithm...")
        start_time = time.time()
        best_state, stats = mcts_engine.optimize(initial_state)
        elapsed = time.time() - start_time
        
        # Print optimization results
        print(f"\nOptimization completed in {elapsed:.2f}s")
        print(f"Iterations: {stats['iterations']}")
        print(f"Tree size: {stats['tree_size']} nodes")
        print(f"Max depth: {stats.get('max_depth', 0)}")
        print(f"Best reward: {stats['best_reward']:.4f}")
        print(f"Evolutionary operations: {stats.get('evolutionary_operations', 0)}")
        print(f"  - Mutations: {stats.get('mutations', 0)}")
        print(f"  - Crossovers: {stats.get('crossovers', 0)}")
        print(f"Error feedback actions: {stats.get('error_feedback_actions', 0)}")
        print(f"Knowledge integrations: {stats.get('knowledge_integrations', 0)}")
        
        # 4. DOMAIN KNOWLEDGE INTEGRATION (already integrated during optimization)
        # ----------------------------
        print("\n[Phase 4] Domain Knowledge Integration (completed during optimization)")
        
        # 5. FINAL PROMPT GENERATION AND OUTPUT
        # -------------------------------------
        print("\n[Phase 5] Final Prompt Generation and Output")
        
        # Process and select optimal output
        print("\n[Phase 5.1] Optimal Prompt Selection and Refinement")
        root_node = mcts_engine._root_node
        optimal_state, selection_stats = self.prompt_selector.select_optimal_prompt(
            root_node, strategy="composite"
        )
        
        print("\n[Phase 5.2] Token Efficiency Optimization")
        final_output, processing_stats = self.output_processor.process_output(
            optimal_state, None, "standard"
        )
        
        print("\n[Phase 5.3] Final Output Processing")
        
        # Save results
        results = {
            "task_name": self.task_name,
            "initial_prompt": initial_prompt,
            "optimized_prompt": optimal_state.text,
            "final_output": final_output,
            "optimization_stats": {
                "iterations": stats['iterations'],
                "time": elapsed,
                "best_reward": stats['best_reward'],
                "tree_size": stats['tree_size'],
                "max_depth": stats.get('max_depth', 0),
                "evolutionary_operations": stats.get('evolutionary_operations', 0),
                "mutations": stats.get('mutations', 0),
                "crossovers": stats.get('crossovers', 0),
                "error_feedback_actions": stats.get('error_feedback_actions', 0),
                "knowledge_integrations": stats.get('knowledge_integrations', 0)
            },
            "selection_stats": selection_stats,
            "processing_stats": processing_stats
        }
        
        # Save to file
        output_path = self.output_dir / "optimization_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        output_text_path = self.output_dir / "optimized_prompt.txt"
        with open(output_text_path, "w") as f:
            f.write(final_output)
        
        print(f"\nResults saved to {self.output_dir}")
        
        print("\nInitial prompt:")
        print("-" * 40)
        print(initial_prompt)
        print("-" * 40)
        
        print("\nOptimized prompt:")
        print("-" * 40)
        print(final_output)
        print("-" * 40)
        
        return results
    
    async def evaluate_final_prompt(self, prompt, num_examples=10):
        """
        Evaluate the final optimized prompt on test data.
        
        Args:
            prompt: The prompt to evaluate.
            num_examples: Number of test examples to use.
            
        Returns:
            Evaluation results.
        """
        print("\nEvaluating optimized prompt on test data...")
        
        # Get test examples - try different ways to get test data
        test_examples = []
        
        # Try different ways to get test examples
        if hasattr(self.task, 'test_set'):
            test_examples = self.task.test_set
        elif hasattr(self.task, 'dataset') and isinstance(self.task.dataset, dict) and 'test' in self.task.dataset:
            test_examples = self.task.dataset['test']
        elif hasattr(self.task, 'test_size'):
            test_examples = self.task.test_size
            
        # Limit to specified number
        if test_examples and num_examples:
            test_examples = test_examples[:num_examples]
        
        if not test_examples:
            print("No test examples available for evaluation. Creating minimal test examples...")
            # Create some minimal test examples for testing
            test_examples = [
                {"question": "Test question 1", "answer": "Test answer 1"},
                {"question": "Test question 2", "answer": "Test answer 2"}
            ]
        
        # Evaluate on test data
        result = await self.evaluate_prompt(prompt, test_examples)
        metrics = result.get("metrics", {})
        
        print(f"Test evaluation results:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"- {metric}: {value:.4f}")
        
        # Save metrics to file
        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a specific task with MCTS-Evo-Prompt")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--iterations", type=int, help="Maximum MCTS iterations (optional)")
    parser.add_argument("--time-limit", type=float, help="Time limit in seconds (optional)")
    parser.add_argument("--data", help="Path to task data (optional)")
    parser.add_argument("--output", help="Output directory (optional)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--test-only", action="store_true", help="Only test connection and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Ensure environment is set up
    setup_environment()
    
    # Create task runner
    try:
        runner = TaskRunner(
            task_name=args.task,
            data_path=args.data,
            llm_model=args.model,
            output_dir=args.output,
            use_gpu=not args.no_gpu,
            debug_mode=args.debug
        )
    except Exception as e:
        print(f"Error initializing TaskRunner: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test connection
    connected = await runner.test_llm_connection()
    if not connected:
        print("Failed to connect to LLM. Exiting.")
        return 1
    
    # Test GPU status
    if not args.no_gpu:
        await runner.check_gpu_status()
    
    if args.test_only:
        return 0
    
    # Get task-specific parameters
    task_params = runner._get_task_specific_params()
    
    # Run optimization with either user-specified or task-specific parameters
    iterations = args.iterations or task_params.get("iterations", 30)
    time_limit = args.time_limit or task_params.get("time_limit", 300)
    
    try:
        results = await runner.run_optimization(
            iterations=iterations,
            time_limit=time_limit
        )
        
        # Evaluate final prompt
        await runner.evaluate_final_prompt(results["final_output"])
        
        return 0
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))