"""
Phase2
Demonstration of the MDP Framework components.
"""
import sys
import os
import time
from colorama import init, Fore, Style

# Add project root directory to Python path
# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# 确保backend目录也在路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from app.core.mdp.state import PromptState
from app.core.mdp.action import (
    AddRoleAction, AddGoalAction, ModifyWorkflowAction, AddConstraintAction,
    AddExplanationAction, AddExampleAction, AddDomainKnowledgeAction,
    SpecifyFormatAction, create_action
)
from app.core.mdp.transition import StateTransition
from app.core.mdp.reward import RewardFunction, PerformanceEvaluator

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

def print_prompt(state):
    """Print a prompt state with highlighting"""
    print(f"{Fore.GREEN}Prompt State ID: {state.state_id[:8]}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Depth: {state.get_depth()}{Style.RESET_ALL}")
    
    if state.action_applied:
        print(f"{Fore.GREEN}Last Action: {state.action_applied}{Style.RESET_ALL}")
    
    print(f"{Fore.WHITE}{state.text}{Style.RESET_ALL}")
    
    # Print metrics
    if state.metrics:
        print(f"\n{Fore.CYAN}Metrics:{Style.RESET_ALL}")
        for key, value in state.metrics.items():
            print(f"  {key}: {value}")

def simulate_typing(text, delay=0.01):
    """Simulate typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo_state_representation():
    """Demonstrate state representation."""
    print_header("State Representation Demo")
    
    # Create a basic state
    basic_text = """
    Role: Sentiment Analysis Expert
    Task: Analyze the sentiment of the provided text.
    
    Steps:
    - Read the text carefully
    - Identify sentiment-bearing words and phrases
    - Determine overall sentiment
    
    Output Format: Provide sentiment as positive, negative, or neutral.
    """
    
    state = PromptState(basic_text)
    
    print_section("Initial State")
    print_prompt(state)
    
    print_section("Components")
    for key, value in state.components.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")
    
    print_section("State Metrics")
    print(f"Structural Completeness: {state.get_structural_completeness():.2f}")
    print(f"Token Efficiency: {state.get_token_efficiency():.2f}")

def demo_action_application():
    """Demonstrate action application."""
    print_header("Action Application Demo")
    
    # Create a minimal state
    minimal_text = """
    Task: Analyze the text.
    """
    
    state = PromptState(minimal_text)
    
    print_section("Initial State")
    print_prompt(state)
    
    # Apply a sequence of actions
    actions = [
        AddRoleAction(parameters={"role_text": "Sentiment Analysis Expert"}),
        ModifyWorkflowAction(parameters={
            "steps": [
                "Read the text carefully",
                "Identify sentiment-bearing words and phrases",
                "Determine overall sentiment"
            ]
        }),
        AddExplanationAction(parameters={
            "explanation_text": "Focus on emotional language and tone",
            "target": "step",
            "target_index": 1
        }),
        AddDomainKnowledgeAction(parameters={
            "knowledge_text": "Sentiment is typically classified as positive, negative, or neutral",
            "domain": "sentiment_analysis"
        }),
        SpecifyFormatAction(parameters={
            "format_text": "Provide the sentiment as 'Sentiment: [positive/negative/neutral]' followed by confidence level and key indicators"
        }),
        AddExampleAction(parameters={
            "example_text": "Input: 'I absolutely love this product!'\nOutput: Sentiment: Positive (Confidence: High)\nKey indicators: 'love', positive exclamation",
            "example_type": "input_output"
        })
    ]
    
    current_state = state
    for i, action in enumerate(actions, 1):
        print_section(f"Action {i}: {action}")
        
        # Apply action
        print("Applying action...")
        time.sleep(0.5)
        new_state = action.apply(current_state)
        
        # Show result
        print_prompt(new_state)
        
        # Update current state
        current_state = new_state

def demo_transition_system():
    """Demonstrate the transition system."""
    print_header("Transition System Demo")
    
    # Create initial state
    initial_text = """
    Task: Analyze the sentiment of the provided text.
    """
    
    initial_state = PromptState(initial_text)
    
    print_section("Initial State")
    print_prompt(initial_state)
    
    # Create transition handler
    transition = StateTransition(stochasticity=0.1)
    
    # Create a sequence of actions
    actions = [
        create_action("add_role", parameters={"role_text": "Sentiment Analysis Expert"}),
        create_action("modify_workflow", parameters={
            "steps": [
                "Read the text carefully",
                "Identify sentiment-bearing words and phrases",
                "Determine overall sentiment"
            ]
        }),
        create_action("add_domain_knowledge", parameters={
            "knowledge_text": "Sentiment analysis involves identifying emotional tone in text",
            "domain": "sentiment_analysis",
            "location": "after_role"
        }),
        create_action("specify_format", parameters={
            "format_text": "Provide sentiment as positive, negative, or neutral with confidence score"
        })
    ]
    
    # Apply transitions
    current_state = initial_state
    states = [current_state]
    
    for i, action in enumerate(actions, 1):
        print_section(f"Transition {i}: {action}")
        
        # Apply transition
        print("Applying transition...")
        time.sleep(0.5)
        new_state = transition.apply(current_state, action)
        
        # Show result
        print_prompt(new_state)
        
        # Update current state and history
        current_state = new_state
        states.append(current_state)
    
    print_section("State History")
    for i, state in enumerate(states):
        print(f"State {i}: ID={state.state_id[:8]}, Depth={state.get_depth()}")

def demo_reward_function():
    """Demonstrate the reward function."""
    print_header("Reward Function Demo")
    
    # Create states with varying quality
    states = [
        # Empty state
        PromptState(""),
        
        # Minimal state
        PromptState("""
        Task: Analyze the sentiment.
        """),
        
        # Basic state
        PromptState("""
        Role: Analyst
        Task: Analyze the sentiment of the text.
        Steps:
        - Read the text
        - Determine sentiment
        """),
        
        # Comprehensive state
        PromptState("""
        Role: Sentiment Analysis Expert
        Task: Analyze the emotional tone of the provided text and classify its sentiment.
        
        Steps:
        - Read the text carefully, identifying sentiment-bearing words and phrases
        - Evaluate the overall sentiment polarity (positive, negative, neutral)
        - Consider the intensity of the expressed sentiment
        - Determine the final classification
        
        Output Format: Provide the sentiment classification as 'Sentiment: [positive/negative/neutral]' 
        followed by a confidence score and a brief explanation.
        
        Example:
        Input: "I absolutely love this product! It exceeds all my expectations."
        Output: Sentiment: Positive
        Confidence: High
        Explanation: Strong positive words ("love", "exceeds expectations") and enthusiastic punctuation.
        """)
    ]
    
    # Create reward functions
    default_reward = RewardFunction()
    
    # Use task-specific evaluator
    sentiment_evaluator = PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
    sentiment_reward = RewardFunction(task_performance_fn=sentiment_evaluator)
    
    print_section("Reward Analysis")
    
    for i, state in enumerate(states):
        print(f"\n{Fore.CYAN}State {i+1}:{Style.RESET_ALL}")
        
        # Calculate rewards
        default_score = default_reward.calculate(state)
        sentiment_score = sentiment_reward.calculate(state)
        
        # Get component rewards
        components = sentiment_reward.get_component_rewards(state)
        
        # Display scores
        print(f"Default Reward: {default_score:.4f}")
        print(f"Sentiment-specific Reward: {sentiment_score:.4f}")
        print("Component Rewards:")
        print(f"  Performance: {components['performance']:.4f}")
        print(f"  Structural: {components['structural_completeness']:.4f}")
        print(f"  Efficiency: {components['token_efficiency']:.4f}")
        
        # Display first few lines of state text
        text_preview = "\n".join(state.text.strip().split("\n")[:3])
        print(f"\nPreview: {text_preview}...")

def demo_action_generation():
    """Demonstrate action generation based on prompt state."""
    print_header("Action Generation Demo")
    
    # Create a state
    state_text = """
    Role: Analyst
    Task: Analyze the customer reviews to determine sentiment.
    
    Steps:
    - Read each review
    - Identify positive and negative expressions
    - Classify the overall sentiment
    """
    
    state = PromptState(state_text)
    
    print_section("Initial State")
    print_prompt(state)
    
    print_section("Potential Actions")
    
    # Generate potential actions based on state analysis
    potential_actions = []
    
    # Check for missing or enhanceable components
    if not state.has_component("output_format"):
        potential_actions.append(
            create_action("specify_format", parameters={
                "format_text": "Provide sentiment as 'Sentiment: [positive/negative/neutral]' with confidence score"
            })
        )
    
    if "Expert" not in state.components.get("role", ""):
        potential_actions.append(
            create_action("add_role", parameters={
                "role_text": "Sentiment Analysis Expert",
                "replace": True
            })
        )
    
    if not state.has_component("examples"):
        potential_actions.append(
            create_action("add_example", parameters={
                "example_text": "Input: 'This product is amazing!'\nOutput: Sentiment: Positive\nConfidence: High",
                "example_type": "input_output"
            })
        )
    
    if not any("domain" in str(state.components) or "knowledge" in str(state.components)):
        potential_actions.append(
            create_action("add_domain_knowledge", parameters={
                "knowledge_text": "Sentiment analysis involves detecting emotional tone through language patterns.",
                "domain": "sentiment_analysis"
            })
        )
    
    # Display potential actions
    for i, action in enumerate(potential_actions, 1):
        print(f"{i}. {action}")
        
        # Simulate applying the action
        new_state = action.apply(state)
        
        # Calculate potential reward improvement
        reward_fn = RewardFunction(
            task_performance_fn=PerformanceEvaluator.get_evaluator_for_task("sentiment_analysis")
        )
        
        current_reward = reward_fn.calculate(state)
        new_reward = reward_fn.calculate(new_state)
        improvement = new_reward - current_reward
        
        print(f"   Reward improvement: {improvement:.4f} ({current_reward:.4f} → {new_reward:.4f})")
        print(f"   Preview: {action.description}")
        print()

def main():
    """Main function"""
    print_header("MDP Framework Demo")
    
    menu_options = [
        "State Representation Demo",
        "Action Application Demo",
        "Transition System Demo", 
        "Reward Function Demo",
        "Action Generation Demo",
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
                demo_state_representation()
            elif choice == 2:
                demo_action_application()
            elif choice == 3:
                demo_transition_system()
            elif choice == 4:
                demo_reward_function()
            elif choice == 5:
                demo_action_generation()
            elif choice == 6:
                # Complete process demo
                demo_state_representation()
                demo_action_application()
                demo_transition_system()
                demo_reward_function()
                demo_action_generation()
            elif choice == 7:
                print("\nThank you for using the MDP Framework Demo!\n")
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