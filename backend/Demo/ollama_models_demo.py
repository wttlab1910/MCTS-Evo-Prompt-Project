"""
Demo application for interacting with Ollama LLM models.

This application provides a command-line interface for testing and 
comparing Mistral, Gemma3, and DeepSeek models using Ollama.
"""
import asyncio
import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent))

from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider

# Available models
MODELS = {
    "mistral": "Mistral",
    "gemma3:12b": "Gemma3 (12B)",
    "deepseek-r1:32b": "DeepSeek-r1 (32B)"
}

async def generate_text(model_name: str, prompt: str, 
                       temperature: float = 0.7, 
                       max_tokens: int = 2048) -> Optional[Dict[str, Any]]:
    """
    Generate text using the specified model.
    
    Args:
        model_name: Name of the model to use
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Response dictionary or None if an error occurred
    """
    try:
        # Create LLM instance
        llm = LLMFactory.create("ollama", model_id=model_name)
        
        # Generate text
        start_time = time.time()
        print(f"Generating with {model_name}...")
        
        response = await llm.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.2f}s")
        
        return response
    except Exception as e:
        print(f"Error generating text with {model_name}: {e}")
        return None

async def compare_models(prompt: str, models: List[str], 
                        temperature: float = 0.7,
                        max_tokens: int = 2048):
    """
    Compare responses from multiple models.
    
    Args:
        prompt: Input prompt
        models: List of model names to compare
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
    """
    print("\n" + "=" * 50)
    print(f"COMPARING MODELS")
    print("=" * 50)
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    # Generate responses from all models
    tasks = [generate_text(model, prompt, temperature, max_tokens) for model in models]
    responses = await asyncio.gather(*tasks)
    
    # Print responses
    for model, response in zip(models, responses):
        if response:
            print("\n" + "-" * 50)
            print(f"Model: {model}")
            print(f"Response time: {response['elapsed_time']:.2f}s")
            print("-" * 50)
            print(response["text"])
        else:
            print("\n" + "-" * 50)
            print(f"Model: {model}")
            print("No response generated")
    
    print("\n" + "=" * 50)

async def interactive_mode():
    """Run interactive mode allowing user to input prompts."""
    print("\n" + "=" * 50)
    print("INTERACTIVE MODE")
    print("=" * 50)
    print("Type 'exit' or 'quit' to exit.")
    print("Available models: " + ", ".join(MODELS.keys()))
    print("=" * 50)
    
    while True:
        # Get model selection
        print("\nSelect model(s) (comma-separated, or 'all'):")
        model_input = input("> ").strip().lower()
        
        if model_input in ["exit", "quit"]:
            break
            
        if model_input == "all":
            selected_models = list(MODELS.keys())
        else:
            selected_models = [m.strip() for m in model_input.split(",")]
            # Validate models
            invalid_models = [m for m in selected_models if m not in MODELS]
            if invalid_models:
                print(f"Invalid models: {', '.join(invalid_models)}")
                continue
        
        # Get prompt
        print("\nEnter your prompt:")
        prompt = input("> ").strip()
        
        if prompt in ["exit", "quit"]:
            break
        
        # Get temperature
        print("\nEnter temperature (0.0-1.0, default 0.7):")
        temp_input = input("> ").strip()
        try:
            temperature = float(temp_input) if temp_input else 0.7
            if not 0 <= temperature <= 1:
                print("Temperature must be between 0 and 1, using default 0.7")
                temperature = 0.7
        except ValueError:
            print("Invalid temperature, using default 0.7")
            temperature = 0.7
        
        # Get max tokens
        print("\nEnter max tokens (default 2048):")
        tokens_input = input("> ").strip()
        try:
            max_tokens = int(tokens_input) if tokens_input else 2048
            if max_tokens < 1:
                print("Max tokens must be positive, using default 2048")
                max_tokens = 2048
        except ValueError:
            print("Invalid max tokens, using default 2048")
            max_tokens = 2048
        
        # Compare models
        await compare_models(prompt, selected_models, temperature, max_tokens)

async def run_demos():
    """Run predefined demo scenarios."""
    # Register Ollama provider if not already registered
    if "ollama" not in LLMFactory._providers:
        LLMFactory.register_provider("ollama", OllamaProvider)
        print("Registered Ollama provider with LLMFactory")
    
    # Demo prompts
    prompts = [
        "Write a 4-line poem about artificial intelligence.",
        "Explain the concept of quantum computing to a 10-year-old.",
        "What are three interesting facts about the history of mathematics?"
    ]
    
    available_models = []
    
    # Check which models are available
    for model_name in MODELS.keys():
        try:
            response = await generate_text(
                model_name, 
                "Hello, are you working?", 
                max_tokens=10
            )
            if response and response.get("text"):
                available_models.append(model_name)
                print(f"✅ {model_name} is available")
            else:
                print(f"❌ {model_name} is not responding properly")
        except Exception as e:
            print(f"❌ {model_name} is not available: {e}")
    
    if not available_models:
        print("No models are available. Please check your Ollama installation.")
        return
    
    # Run demos
    for i, prompt in enumerate(prompts):
        print(f"\n\nDEMO {i+1}/{len(prompts)}")
        await compare_models(prompt, available_models)
    
    # Start interactive mode
    await interactive_mode()

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ollama Models Demo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt to use")
    parser.add_argument("--model", "-m", type=str, default="all", help="Model to use (comma-separated, or 'all')")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", "-mt", type=int, default=2048, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Register Ollama provider if not already registered
    if "ollama" not in LLMFactory._providers:
        LLMFactory.register_provider("ollama", OllamaProvider)
        print("Registered Ollama provider with LLMFactory")
    
    if args.interactive:
        await interactive_mode()
    elif args.prompt:
        # Use specified model(s)
        if args.model == "all":
            models = list(MODELS.keys())
        else:
            models = [m.strip() for m in args.model.split(",")]
        
        await compare_models(args.prompt, models, args.temperature, args.max_tokens)
    else:
        # Run demos
        await run_demos()

if __name__ == "__main__":
    asyncio.run(main())