# """
# Example script demonstrating the use of the Input Processing phase.
# """
import os
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.core.input.input_processor import InputProcessor

# Create input processor
processor = InputProcessor()

# Example inputs
examples = [
    "Classify the sentiment of this review: The movie was absolutely fantastic, with stunning visuals and an engaging plot.",
    "Instruction: Extract all person names from the text. Data: John Smith met with Sarah Johnson and Dr. Robert Brown to discuss the project.",
    "Generate a short story about a robot learning to paint.",
    "What is machine learning and how does it work?"
]

# Process each example
print("Processing example inputs...\n")

for i, example in enumerate(examples, 1):
    print(f"Example {i}: {example[:50]}...")
    result = processor.process_input(example)
    
    print(f"  Task Type: {result['task_analysis']['task_type']}")
    print(f"  Category: {result['task_analysis']['category']}")
    print(f"  Domain: {result['domain']}")
    print(f"  Confidence: {result['task_analysis']['confidence']}")
    print(f"  Expanded Prompt Length: {len(result['expanded_prompt'])} chars")
    print(f"  Original Instruction Length: {len(result['instruction'])} chars")
    print(f"  Expansion Ratio: {len(result['expanded_prompt']) / len(result['instruction']):.2f}x")
    print("\n  Expanded Prompt Preview (first 200 chars):")
    print(f"  {result['expanded_prompt'][:200]}...")
    print("\n" + "-" * 80 + "\n")

# Save the last result to a JSON file for inspection
output_dir = "examples/output"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "processed_example.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Full result saved to {os.path.join(output_dir, 'processed_example.json')}")