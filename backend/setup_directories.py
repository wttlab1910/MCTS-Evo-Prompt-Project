"""
Create the necessary directory structure for MCTS-Evo-Prompt.
"""
import os
import argparse
from pathlib import Path

def create_directory_structure(base_dir=None):
    """
    Create the directory structure for MCTS-Evo-Prompt.
    
    Args:
        base_dir: Base directory for the project.
    """
    if base_dir:
        base_dir = Path(base_dir)
    else:
        # Use current directory as base
        base_dir = Path.cwd()
    
    # Create directories
    directories = [
        # App directories
        base_dir / "app" / "core" / "mcts",
        base_dir / "app" / "core" / "mdp",
        base_dir / "app" / "core" / "evolution",
        base_dir / "app" / "core" / "optimization",
        base_dir / "app" / "core" / "input",
        base_dir / "app" / "tasks",
        base_dir / "app" / "llm" / "providers",
        base_dir / "app" / "llm" / "evaluation",
        base_dir / "app" / "knowledge" / "error",
        base_dir / "app" / "services",
        base_dir / "app" / "utils",
        
        # Data directories
        base_dir / "app" / "data" / "cached",
        base_dir / "app" / "data" / "knowledge_base" / "prompt_guide" / "techniques",
        base_dir / "app" / "data" / "knowledge_base" / "prompt_guide" / "templates",
        base_dir / "app" / "data" / "knowledge_base" / "prompt_guide" / "examples",
        base_dir / "app" / "data" / "tasks",
        base_dir / "app" / "data" / "logs",
        base_dir / "app" / "data" / "models",
        
        # Output directory
        base_dir / "output"
    ]
    
    # Create directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create placeholder files to ensure git tracks empty directories
    for directory in directories:
        placeholder = directory / ".placeholder"
        if not placeholder.exists():
            with open(placeholder, "w") as f:
                f.write("# This file ensures the directory is tracked by git\n")
            print(f"Created placeholder: {placeholder}")
    
    print(f"Directory structure created successfully at {base_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create directory structure for MCTS-Evo-Prompt")
    parser.add_argument("--base-dir", help="Base directory for the project")
    
    args = parser.parse_args()
    create_directory_structure(args.base_dir)

if __name__ == "__main__":
    main()