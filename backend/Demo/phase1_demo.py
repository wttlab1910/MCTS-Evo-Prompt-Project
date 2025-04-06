
"""
MCTS-Evo-Prompt Phase 1 Demo Program
This program demonstrates the functionality of input processing and initialization modules, including:
- Prompt and data separation
- Task type analysis
- Task understanding
- Prompt expansion
- Application of prompt engineering guidelines
"""
import sys
import os
import time
from colorama import init, Fore, Style

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from app.core.input.prompt_separator import PromptSeparator
from app.core.input.task_analyzer import TaskAnalyzer
from app.core.input.prompt_expander import PromptExpander
from app.core.input.prompt_guide_loader import PromptGuideLoader
from app.services.prompt_service import PromptService

# Initialize colorama
init()

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

def print_json(obj, indent=2):
    """Pretty print JSON object"""
    import json
    print(json.dumps(obj, indent=indent, ensure_ascii=False))

def simulate_typing(text, delay=0.01):
    """Simulate typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo_separator():
    """Demonstrate prompt separator functionality"""
    print_header("Prompt and Data Separation Demo")
    separator = PromptSeparator()
    
    examples = [
        "Instruction: Analyze the sentiment of this text. Data: This product exceeded my expectations, very satisfied!",
        "Please summarize the following article.\n\nArtificial Intelligence (AI) is developing rapidly, affecting various industries. Machine learning technology enables computers to learn from data and improve. Deep learning further advances progress in image recognition and natural language processing. Nevertheless, AI ethics issues are increasingly raising concerns.",
        "Extract entities from: John visited Paris with his friend Sarah in June 2023."
    ]
    
    for i, example in enumerate(examples, 1):
        print_section(f"Example {i}")
        print(f"Original input: {example}")
        
        prompt, data = separator.separate(example)
        
        print_result("Prompt part", prompt)
        print_result("Data part", data)
        print()

def demo_analyzer():
    """Demonstrate task analyzer functionality"""
    print_header("Task Type Analysis Demo")
    analyzer = TaskAnalyzer()
    
    examples = [
        "Classify this review as positive or negative.",
        "Summarize this article in three sentences.",
        "Extract the main entities from this text.",
        "Write a Python function to sort a list.",
        "Determine if the sentiment is positive or negative.",
        "Translate this text to French.",
        "What are the causes of global warming?"
    ]
    
    for i, example in enumerate(examples, 1):
        print_section(f"Example {i}: {example}")
        
        # Simulate analysis process
        print("Analyzing...", end="", flush=True)
        time.sleep(0.5)
        
        analysis = analyzer.analyze(example)
        print("\r", end="")
        
        print_result("Task type", f"{analysis['task_type']} (Confidence: {analysis['task_confidence']:.2f})")
        print_result("Task subcategory", analysis['category'])
        
        if analysis['key_concepts']:
            print_result("Key concepts", ", ".join(analysis['key_concepts']))
            
        print_result("Suggested evaluation methods", ", ".join(analysis['evaluation_methods']))
        print()

def demo_expander():
    """Demonstrate prompt expander functionality"""
    print_header("Prompt Expansion Demo")
    analyzer = TaskAnalyzer()
    expander = PromptExpander()
    
    examples = [
        "Summarize this medical research paper.",
        "Classify this email as spam or not spam.",
        "Extract the dates and locations from this text."
    ]
    
    for i, example in enumerate(examples, 1):
        print_section(f"Example {i}: {example}")
        
        # Analyze task
        analysis = analyzer.analyze(example)
        print_result("Identified task type", analysis['task_type'])
        
        # Expand prompt
        print("Expanding prompt...", end="", flush=True)
        time.sleep(0.8)
        print("\r", end="")
        
        analysis["prompt"] = example  # Ensure prompt field exists
        expanded = expander.expand(example, analysis)
        
        print(f"{Fore.GREEN}Expanded prompt:{Style.RESET_ALL}")
        simulate_typing(expanded, 0.001)
        print()

def demo_guide_loader():
    """Demonstrate prompt engineering guide loader functionality"""
    print_header("Prompt Engineering Guidelines Demo")
    guide_loader = PromptGuideLoader()
    
    # Ensure default guide exists
    guide_loader.create_default_guide()
    
    # Get available techniques and templates
    techniques = guide_loader.get_all_techniques()
    task_types = guide_loader.get_all_task_types()
    
    print_section("Available Prompt Techniques")
    print(", ".join(techniques))
    
    print_section("Supported Task Types")
    print(", ".join(task_types))
    
    # Show technique details
    if "zero_shot" in techniques:
        print_section("Zero-Shot Prompt Technique")
        tech = guide_loader.get_technique("zero_shot")
        print_result("Name", tech['name'])
        print_result("Description", tech['description'])
        print_result("Key concepts", ", ".join(tech['key_concepts']))
        print_result("Number of examples", len(tech['examples']))
        print_result("Best practices", tech['best_practices'][0])
    
    # Show template details
    if "classification" in task_types:
        print_section("Classification Task Template")
        template = guide_loader.get_template("classification")
        print_result("Task type", template['task_type'])
        print_result("Number of templates", len(template['templates']))
        print_result("Example template name", template['templates'][0]['name'])
    
    # Generate training data
    print_section("Training Data Generation")
    training_data = guide_loader.generate_training_data()
    print_result("Number of generated training samples", len(training_data))
    if training_data:
        print_section("Training Data Example")
        print_result("Input", training_data[0]['input'])
        print_result("Target", training_data[0]['target'][:100] + "...")

def demo_service():
    """Demonstrate complete service functionality"""
    print_header("End-to-End Processing Demo")
    service = PromptService()
    
    while True:
        print_section("Please enter prompt and data (enter 'exit' to quit)")
        print("Example format: 'Instruction: Classify this review. Data: This product is great!'")
        print("Or just enter a prompt, like: 'Extract key people and dates from this text'")
        
        user_input = input(f"\n{Fore.CYAN}> {Style.RESET_ALL}")
        
        if user_input.lower() in ('exit', 'quit', 'q'):
            break
        
        if not user_input.strip():
            continue
        
        # Process input
        print("\nProcessing...", end="", flush=True)
        time.sleep(0.8)
        print("\r", end="")
        
        result = service.process_input(user_input)
        
        # Show results
        print_section("Separation Results")
        print_result("Prompt part", result['prompt'])
        print_result("Data part", result['data'])
        
        print_section("Task Analysis")
        print_result("Task type", result['task_analysis']['task_type'])
        print_result("Task confidence", f"{result['task_analysis']['task_confidence']:.2f}")
        print_result("Task category", result['task_analysis']['category'])
        
        if result['task_analysis']['key_concepts']:
            print_result("Key concepts", ", ".join(result['task_analysis']['key_concepts']))
        
        print_section("Expanded Prompt")
        simulate_typing(result['expanded_prompt'], 0.001)

def main():
    """Main function"""
    print_header("MCTS-Evo-Prompt Phase 1 Demo")
    
    menu_options = [
        "Prompt and Data Separation Demo",
        "Task Type Analysis Demo",
        "Prompt Expansion Demo",
        "Prompt Engineering Guidelines Demo",
        "End-to-End Processing Demo",
        "Complete Process Demo",
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
                demo_separator()
            elif choice == 2:
                demo_analyzer()
            elif choice == 3:
                demo_expander()
            elif choice == 4:
                demo_guide_loader()
            elif choice == 5:
                demo_service()
            elif choice == 6:
                # Complete process demo
                demo_separator()
                demo_analyzer()
                demo_expander()
                demo_guide_loader()
                demo_service()
            elif choice == 7:
                print("\nThank you for using MCTS-Evo-Prompt Phase 1 Demo Program!\n")
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