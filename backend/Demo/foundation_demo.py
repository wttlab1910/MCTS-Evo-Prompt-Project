"""
MCTS-Evo-Prompt Foundation Components Demo
This program demonstrates the functionality of the foundation components, including:
- LLM Interface with Ollama Models
- Utilities (Logging, Caching, Timing)
- Services (Prompt, Optimization, Knowledge)
- API Features
"""
import sys
import os
import time
import asyncio
from colorama import init, Fore, Style

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider
from app.llm.evaluation.metrics import EvaluationMetrics
from app.llm.evaluation.validator import ResponseValidator
from app.services.prompt_service import PromptService
from app.services.optimization_service import OptimizationService
from app.services.knowledge_service import KnowledgeService
from app.utils.logger import get_logger
from app.utils.timer import Timer, timing_stats
from app.utils.cache import MemoryCache
from app.config import LLM_CONFIG

# Initialize colorama
init()

logger = get_logger("foundation_demo")

def print_header(text):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 60}")
    print(f"{text.center(60)}")
    print(f"{'=' * 60}{Style.RESET_ALL}")

def print_section(text):
    """Print colored section title"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_result(label, content):
    """Print result label and content"""
    print(f"{Fore.GREEN}{label}:{Style.RESET_ALL} {content}")

def simulate_typing(text, delay=0.01):
    """Simulate typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

async def demo_ollama_models():
    """Demonstrate Ollama models functionality."""
    print_header("Ollama Models Demo")
    
    print_section("Available Ollama Models")
    print(f"Default model: mistral")
    
    # Register Ollama provider if not already registered
    if "ollama" not in LLMFactory._providers:
        LLMFactory.register_provider("ollama", OllamaProvider)
        print("Registered Ollama provider with LLMFactory")
    
    models = ["mistral", "gemma3:12b", "deepseek-r1:32b"]
    
    for model_id in models:
        try:
            print(f"\nTesting {model_id}...")
            
            # Create LLM instance
            llm = LLMFactory.create("ollama", model_id=model_id)
            
            # Generate text
            prompt = "Write a short poem about artificial intelligence in exactly 4 lines."
            
            with Timer(f"ollama_{model_id}_generation", log_level="info"):
                response = await llm.generate(prompt)
            
            if response.get("text"):
                print_result("Status", f"{Fore.GREEN}Available{Style.RESET_ALL}")
                print_result("Prompt", prompt)
                print_result("Response", response.get("text", "No response generated"))
                print_result("Model", response.get("model", "unknown"))
                print_result("Elapsed time", f"{response.get('elapsed_time', 0):.3f} seconds")
                print_result("Finish reason", response.get("finish_reason", "unknown"))
            else:
                print_result("Status", f"{Fore.RED}Not Available{Style.RESET_ALL}")
                print_result("Error", response.get("error", "Unknown error"))
        except Exception as e:
            print_result("Status", f"{Fore.RED}Not Available{Style.RESET_ALL}")
            print_result("Error", str(e))
    
    print("\nCheck Ollama installation if any models are not available")
    print("Install models with: ollama pull <model_name>")

async def demo_llm_interface():
    """Demonstrate LLM interface functionality."""
    print_header("LLM Interface Demo")
    
    print_section("Available Providers")
    print(f"Default provider: {LLM_CONFIG['default_provider']}")
    print(f"Available providers: {', '.join(LLM_CONFIG['providers'].keys())}")
    
    # Add Ollama provider if not already registered
    if "ollama" not in LLMFactory._providers:
        LLMFactory.register_provider("ollama", OllamaProvider)
        print("Registered Ollama provider with LLMFactory")
    
    try:
        print_section("Creating LLM Instance")
        provider = "ollama"  # Using Ollama instead of default
        model_id = "mistral"  # Using Mistral model
        
        print(f"Creating {provider} instance with model: {model_id}")
        llm = LLMFactory.create(
            provider=provider,
            model_id=model_id
        )
        
        print(f"{Fore.GREEN}Successfully created LLM instance{Style.RESET_ALL}")
        
        print_section("Text Generation Test")
        print("Generating response (this may take a few seconds)...")
        
        prompt = "Explain the concept of prompt engineering in one sentence."
        
        try:
            with Timer("llm_generation", log_level="info"):
                response = await llm.generate(prompt)
            
            print_result("Prompt", prompt)
            print_result("Response", response.get("text", "No response generated"))
            print_result("Model", response.get("model", "unknown"))
            print_result("Elapsed time", f"{response.get('elapsed_time', 0):.3f} seconds")
            print_result("Finish reason", response.get("finish_reason", "unknown"))
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {e}{Style.RESET_ALL}")
            print("This may be due to missing model files or API keys.")
            print("You can still proceed with the rest of the demo.")
        
        print_section("Batch Processing")
        prompts = [
            "What is machine learning?",
            "Define neural networks."
        ]
        
        print("Generating batch responses (this may take a few seconds)...")
        
        try:
            with Timer("llm_batch_generation", log_level="info"):
                batch_responses = await llm.generate_batch(prompts)
            
            print(f"Generated {len(batch_responses)} responses")
            for i, response in enumerate(batch_responses):
                print(f"\n{Fore.CYAN}Response {i+1}:{Style.RESET_ALL}")
                print(f"Prompt: {response.get('prompt', 'unknown')}")
                print(f"Response: {response.get('text', 'No response')}")
        except Exception as e:
            print(f"{Fore.RED}Error generating batch responses: {e}{Style.RESET_ALL}")
            print("You can still proceed with the rest of the demo.")
    
    except Exception as e:
        print(f"{Fore.RED}Error initializing LLM: {e}{Style.RESET_ALL}")
        print("This may be due to missing dependencies or configurations.")
        print("You can still proceed with the rest of the demo.")
    
    print_section("Evaluation Metrics")
    prediction = "This product is great! I highly recommend it."
    reference = "Positive sentiment. Highly recommended product."
    
    metrics = {
        "Exact Match": EvaluationMetrics.exact_match(prediction, reference),
        "F1 Score": EvaluationMetrics.f1_score(prediction, reference),
        "Token Accuracy": EvaluationMetrics.token_accuracy(prediction, reference),
        "Containment Score": EvaluationMetrics.containment_score(prediction, reference)
    }
    
    print_result("Prediction", prediction)
    print_result("Reference", reference)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Show task-specific evaluations
    task_evaluations = {
        "Classification": EvaluationMetrics.evaluate_classification("Positive", "Positive"),
        "Extraction": EvaluationMetrics.evaluate_extraction("John, Sarah", "John, Sarah, Michael"),
        "Generation": EvaluationMetrics.evaluate_generation("Summary of article", "This is a summary of the article about...")
    }
    
    print("\nTask-specific evaluations:")
    for task, result in task_evaluations.items():
        print(f"  {task}: {result['overall'] if 'overall' in result else list(result.values())[0]:.4f}")
    
    print_section("Response Validation")
    validator = ResponseValidator(task_type="classification")
    
    response = {
        "text": "Positive",
        "finish_reason": "stop",
        "elapsed_time": 0.5
    }
    
    validation = validator.validate(response, expected="Positive")
    
    print_result("Valid", validation["valid"])
    print_result("Quality score", f"{validation.get('quality_score', 0):.4f}")
    
    if validation.get("errors"):
        print_result("Errors", ", ".join(validation["errors"]))
    
    if validation.get("warnings"):
        print_result("Warnings", ", ".join(validation["warnings"]))

def demo_utilities():
    """Demonstrate utility functionality."""
    print_header("Utilities Demo")
    
    print_section("Logging")
    logger = get_logger("demo_logger")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print(f"{Fore.GREEN}Logs are written to: {Fore.CYAN}data/logs/demo_logger.log{Style.RESET_ALL}")
    
    print_section("Timing")
    with Timer("operation1", log_level="info") as timer:
        print("Performing operation...")
        time.sleep(0.5)
    
    print_result("Elapsed time", f"{timer.elapsed:.3f} seconds")
    print_result("Elapsed time (ms)", f"{timer.elapsed_ms:.2f} ms")
    
    # Record multiple timings
    timing_stats.record("demo_operation", 0.1)
    timing_stats.record("demo_operation", 0.2)
    timing_stats.record("demo_operation", 0.3)
    
    stats = timing_stats.get_stats("demo_operation")
    
    print("\nOperation statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Min: {stats['min']:.3f}s")
    print(f"  Max: {stats['max']:.3f}s")
    print(f"  Average: {stats['avg']:.3f}s")
    print(f"  Total: {stats['total']:.3f}s")
    
    print_section("Caching")
    cache = MemoryCache(expiration=60)
    
    # Store in cache
    cache.set("demo_key", "Demo value")
    cached_value = cache.get("demo_key")
    
    print_result("Cached value", cached_value)
    
    # Delete from cache
    cache.delete("demo_key")
    deleted_value = cache.get("demo_key")
    
    print_result("After deletion", deleted_value or "None")
    
    # Demonstrate expiration
    cache.set("expiring_key", "Expires quickly", expiration=3)
    print(f"Value will expire in 3 seconds: {cache.get('expiring_key')}")
    
    print("\nChecking after 1 second...")
    time.sleep(1)
    print_result("Value after 1s", cache.get("expiring_key") or "None")
    
    print("Checking after 3 more seconds...")
    time.sleep(3)
    print_result("Value after 4s", cache.get("expiring_key") or "None (expired)")

async def demo_services():
    """Demonstrate service functionality."""
    print_header("Services Demo")
    
    print_section("Prompt Service")
    prompt_service = PromptService()
    
    # Make sure Ollama provider is registered
    if "ollama" not in LLMFactory._providers:
        LLMFactory.register_provider("ollama", OllamaProvider)
        print("Registered Ollama provider with LLMFactory")
    
    # Process input
    input_text = "Instruction: Classify the sentiment of this review. Data: This product is amazing, I love it!"
    print_result("Input", input_text)
    
    with Timer("process_input", log_level="info"):
        result = prompt_service.process_input(input_text)
    
    print("\nProcessing results:")
    print_result("Prompt", result["prompt"])
    print_result("Data", result["data"])
    print_result("Task type", result["task_analysis"]["task_type"])
    print_result("Task confidence", f"{result['task_analysis']['task_confidence']:.2f}")
    
    print("\nExpanded prompt:")
    simulate_typing(result["expanded_prompt"], 0.001)
    
    print_section("Knowledge Service")
    knowledge_service = KnowledgeService()
    
    # List existing entries
    entries = await knowledge_service.list_entries()
    print_result("Existing entries", len(entries))
    
    # Create a new entry
    print("\nCreating new knowledge entry...")
    entry = await knowledge_service.create_entry(
        knowledge_type="concept_definition",
        statement="Prompt engineering is the process of designing effective prompts for language models",
        domain="nlp",
        metadata={"source": "demo", "confidence": 0.95}
    )
    
    print_result("Created entry ID", entry["id"])
    print_result("Type", entry["knowledge_type"])
    print_result("Statement", entry["statement"])
    print_result("Domain", entry["domain"])
    print_result("Created at", entry["created_at"])
    
    # Update the entry
    print("\nUpdating entry...")
    updated = await knowledge_service.update_entry(
        entry_id=entry["id"],
        knowledge_type="concept_definition",
        statement="Prompt engineering is the process of designing effective prompts for large language models",
        domain="nlp",
        metadata={"source": "demo", "confidence": 0.98, "updated": True}
    )
    
    print_result("Updated statement", updated["statement"])
    print_result("Updated confidence", f"{updated['metadata']['confidence']}")
    print_result("Updated at", updated["updated_at"])
    
    # Get entries by domain
    nlp_entries = await knowledge_service.list_entries(domain="nlp")
    print_result("\nNLP domain entries", len(nlp_entries))
    
    # Delete the entry
    print("\nDeleting entry...")
    deleted = await knowledge_service.delete_entry(entry["id"])
    print_result("Deleted", deleted)
    
    # Verify deletion
    verification = await knowledge_service.get_entry(entry["id"])
    print_result("Entry after deletion", verification or "None (deleted)")
    
    print_section("Optimization Service")
    optimization_service = OptimizationService()
    
    # Start optimization
    print("\nStarting prompt optimization...")
    optimization_id = await optimization_service.start_optimization(
        input_text="Instruction: Classify the sentiment of this review. Data: This product is amazing!",
        expected_output="Positive",
        iterations=10,
        timeout=5
    )
    
    print_result("Optimization ID", optimization_id)
    
    # Check status
    print("\nChecking status (initial)...")
    status = await optimization_service.get_optimization_status(optimization_id)
    print_result("Status", status["status"])
    print_result("Progress", f"{status['progress'] * 100:.1f}%")
    
    # Wait for progress
    await asyncio.sleep(2)
    
    # Check status again
    print("\nChecking status (after 2s)...")
    status = await optimization_service.get_optimization_status(optimization_id)
    print_result("Status", status["status"])
    print_result("Progress", f"{status['progress'] * 100:.1f}%")
    
    # Wait for completion or show how to cancel
    if status["status"] == "running":
        print("\nOptimization still running, you can:")
        print("1. Wait for completion")
        print("2. Cancel optimization")
        
        choice = input("\nEnter your choice (1/2): ")
        
        if choice == "2":
            # Cancel optimization
            cancelled = await optimization_service.cancel_optimization(optimization_id)
            print_result("Cancelled", cancelled)
        else:
            # Wait for completion (max 10 more seconds)
            for _ in range(5):
                await asyncio.sleep(2)
                status = await optimization_service.get_optimization_status(optimization_id)
                print_result("Status", status["status"])
                print_result("Progress", f"{status['progress'] * 100:.1f}%")
                
                if status["status"] != "running":
                    break
    
    # Final status
    print("\nFinal optimization status...")
    status = await optimization_service.get_optimization_status(optimization_id)
    print_result("Status", status["status"])
    
    if status["status"] == "completed" and status.get("result"):
        print("\nOptimization result:")
        print_result("Baseline prompt", status["result"]["baseline_prompt"])
        print_result("Optimized prompt", status["result"]["optimized_prompt"])
        print_result("Improvement", f"{status['result']['improvement']:.4f}")

async def main():
    """Main function"""
    print_header("MCTS-Evo-Prompt Foundation Components Demo")
    
    menu_options = [
        "Ollama Models Demo",
        "LLM Interface Demo",
        "Utilities Demo",
        "Services Demo",
        "Complete Demo (All Components)",
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
                await demo_ollama_models()
            elif choice == 2:
                await demo_llm_interface()
            elif choice == 3:
                demo_utilities()
            elif choice == 4:
                await demo_services()
            elif choice == 5:
                # Complete demo
                await demo_ollama_models()
                await demo_llm_interface()
                demo_utilities()
                await demo_services()
            elif choice == 6:
                print("\nThank you for using the Foundation Components Demo!\n")
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
    asyncio.run(main())