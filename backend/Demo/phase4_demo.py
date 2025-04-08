"""
Phase 4: Domain Knowledge Integration Demo
Demonstration of knowledge extraction, verification, and integration.
"""
import sys
import os
import time
import random
from colorama import init, Fore, Style
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Ensure backend directory is also in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from app.knowledge.extraction.extractor import (
    ErrorBasedExtractor,
    ConceptualKnowledgeExtractor,
    ProceduralKnowledgeExtractor
)
from app.knowledge.extraction.verification import (
    ConsistencyVerifier,
    RelationshipMapper,
    ConfidenceScorer
)
from app.knowledge.integration.integrator import PromptKnowledgeIntegrator
from app.knowledge.integration.strategy import (
    PlacementStrategy,
    FormatSelectionStrategy,
    ConflictResolutionStrategy
)
from app.core.mdp.state import PromptState
from app.knowledge.knowledge_base import KnowledgeBase
from app.knowledge.domain.domain_knowledge import DomainKnowledgeManager

# Initialize colorama
init()

def print_header(text):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 80}")
    print(f"{text.center(80)}")
    print(f"{'=' * 80}{Style.RESET_ALL}")

def print_section(text):
    """Print colored section title"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_json(obj, indent=2):
    """Pretty print JSON object"""
    import json
    json_str = json.dumps(obj, indent=indent, ensure_ascii=False)
    
    # Add some color
    json_str = json_str.replace('"id":', f'{Fore.GREEN}"id":{Style.RESET_ALL}')
    json_str = json_str.replace('"type":', f'{Fore.GREEN}"type":{Style.RESET_ALL}')
    json_str = json_str.replace('"statement":', f'{Fore.GREEN}"statement":{Style.RESET_ALL}')
    json_str = json_str.replace('"confidence":', f'{Fore.GREEN}"confidence":{Style.RESET_ALL}')
    
    print(json_str)

def print_prompt(state):
    """Print a prompt state with highlighting"""
    print(f"{Fore.GREEN}Prompt State ID: {state.state_id[:8]}{Style.RESET_ALL}")
    
    # Print text with some formatting
    lines = state.text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("Role:"):
            print(f"{Fore.MAGENTA}{line}{Style.RESET_ALL}")
        elif line.startswith("Task:"):
            print(f"{Fore.BLUE}{line}{Style.RESET_ALL}")
        elif line.startswith("Steps:") or line.startswith("Step "):
            print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
        elif line.startswith("-"):
            print(f"{Fore.CYAN}  {line}{Style.RESET_ALL}")
        elif line.startswith("Domain Knowledge:") or line.startswith("Knowledge:"):
            print(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
        elif line.startswith("Output Format:"):
            print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
        else:
            print(f"{Fore.WHITE}{line}{Style.RESET_ALL}")

def print_relationship_graph(knowledge_items, title="Knowledge Relationship Graph"):
    """Print a simple text-based graph of knowledge relationships"""
    if not knowledge_items:
        print(f"{Fore.RED}No knowledge items to display{Style.RESET_ALL}")
        return
        
    print(f"\n{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    
    # Create simplified relationship view
    entities = set()
    relationships = []
    
    for item in knowledge_items:
        item_id = item.get("id", "unknown")[:8]  # Truncate ID for display
        item_entities = item.get("entities", [])
        
        # Add entities
        for entity in item_entities:
            entities.add(entity)
            relationships.append((item_id, "has_entity", entity))
        
        # Add relations
        for rel in item.get("relations", []):
            subj = rel.get("subject", "")
            pred = rel.get("predicate", "")
            obj = rel.get("object", "")
            
            if subj and pred and obj:
                entities.add(subj)
                entities.add(obj)
                relationships.append((subj, pred, obj))
                relationships.append((item_id, "defines_relation", f"{subj}-{pred}-{obj}"))
    
    # Print entities
    print(f"{Fore.GREEN}Entities:{Style.RESET_ALL}")
    for entity in sorted(entities):
        print(f"  • {entity}")
    
    # Print relationships
    print(f"\n{Fore.GREEN}Relationships:{Style.RESET_ALL}")
    for src, rel_type, dst in sorted(relationships):
        print(f"  {Fore.CYAN}{src}{Style.RESET_ALL} --[{rel_type}]--> {Fore.YELLOW}{dst}{Style.RESET_ALL}")

def simulate_typing(text, delay=0.01):
    """Simulate typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo_knowledge_extraction():
    """Demonstrate knowledge extraction."""
    print_header("Knowledge Extraction Demo")
    
    print_section("1. Error-Based Knowledge Extraction")
    
    # Sample error data
    error_data = [
        {
            "example_id": "e1",
            "example": {
                "text": "The patient has elevated HbA1c levels consistent with diabetes.",
                "expected": "disease_mention"
            },
            "error_type": "entity_confusion",
            "actual": "lab_result",
            "description": "HbA1c was classified as a disease instead of a lab test."
        },
        {
            "example_id": "e2",
            "example": {
                "text": "The treatment plan includes ACE inhibitors for hypertension.",
                "expected": "drug_class_mention",
                "actual": "specific_drug"
            },
            "error_type": "entity_confusion",
            "description": "ACE inhibitors were classified incorrectly as a specific drug instead of a drug class."
        }
    ]
    
    print("Sample error data:")
    print_json(error_data)
    
    # Extract knowledge
    print("\nExtracting knowledge from errors...")
    time.sleep(1)
    
    extractor = ErrorBasedExtractor()
    knowledge_items = extractor.extract(error_data, domain="medical")
    
    print(f"\nExtracted {len(knowledge_items)} knowledge items:")
    for item in knowledge_items:
        print_json(item)
    
    print_section("2. Conceptual Knowledge Extraction")
    
    # Sample text with conceptual knowledge
    medical_text = """
    Angiotensin-converting enzyme (ACE) inhibitors are a class of medication used primarily 
    for the treatment of hypertension and heart failure. They work by inhibiting the 
    angiotensin-converting enzyme, which is responsible for converting angiotensin I to 
    angiotensin II, a potent vasoconstrictor.
    
    HbA1c is defined as glycated hemoglobin, and it is used as a marker of average blood 
    glucose levels over the previous 2-3 months. It is commonly used in the diagnosis and 
    monitoring of diabetes mellitus.
    """
    
    print("Sample text:")
    print(medical_text)
    
    # Extract knowledge
    print("\nExtracting conceptual knowledge...")
    time.sleep(1)
    
    extractor = ConceptualKnowledgeExtractor()
    concept_items = extractor.extract(medical_text, domain="medical")
    
    print(f"\nExtracted {len(concept_items)} conceptual knowledge items:")
    for item in concept_items:
        print_json(item)
    
    print_section("3. Procedural Knowledge Extraction")
    
    # Sample text with procedural knowledge
    procedure_text = """
    How to calculate the Glomerular Filtration Rate (GFR):
    
    Step 1: Gather the necessary patient information (age, sex, race, and serum creatinine level).
    Step 2: Choose the appropriate GFR estimation equation (MDRD or CKD-EPI).
    Step 3: Input the values into the selected equation.
    Step 4: Calculate the estimated GFR.
    Step 5: Interpret the result according to clinical guidelines.
    
    The CKD-EPI equation generally provides a more accurate estimate than the MDRD equation, 
    especially at higher GFR values.
    """
    
    print("Sample procedure text:")
    print(procedure_text)
    
    # Extract knowledge
    print("\nExtracting procedural knowledge...")
    time.sleep(1)
    
    extractor = ProceduralKnowledgeExtractor()
    procedure_items = extractor.extract(procedure_text, domain="medical")
    
    print(f"\nExtracted {len(procedure_items)} procedural knowledge items:")
    for item in procedure_items:
        print_json(item)
    
    # Return all extracted items for use in other demos
    return knowledge_items + concept_items + procedure_items

def demo_knowledge_verification(knowledge_items):
    """Demonstrate knowledge verification."""
    print_header("Knowledge Verification Demo")
    
    if not knowledge_items:
        print(f"{Fore.RED}No knowledge items to verify. Run extraction demo first.{Style.RESET_ALL}")
        return knowledge_items
    
    print_section("1. Consistency Verification")
    
    # Create a sample contradictory item
    contradictory_item = {
        "id": "k_contra",
        "type": "conceptual_knowledge",
        "statement": "HbA1c is a specific type of diabetes, not a lab test.",
        "entities": ["HbA1c"],
        "relations": [
            {"subject": "HbA1c", "predicate": "isA", "object": "disease"}
        ],
        "metadata": {
            "source": "text_extraction",
            "domain": "medical",
            "confidence": 0.6
        }
    }
    
    print("Checking for contradictions with existing knowledge...")
    time.sleep(1)
    
    verifier = ConsistencyVerifier()
    verified = verifier.verify(contradictory_item, existing_knowledge=knowledge_items)
    
    print("\nVerification result:")
    print_json(verified["metadata"]["verification"])
    
    # Create a sample duplicate item
    duplicate_item = knowledge_items[0].copy()
    duplicate_item["id"] = "k_dup"
    duplicate_item["statement"] = duplicate_item["statement"] + " This is important to remember."
    
    print("\nChecking for duplicates...")
    time.sleep(1)
    
    verified_dup = verifier.verify(duplicate_item, existing_knowledge=knowledge_items)
    
    print("\nDuplicate verification result:")
    print_json(verified_dup["metadata"]["verification"])
    
    print_section("2. Relationship Mapping")
    
    print("Mapping relationships between knowledge items...")
    time.sleep(1)
    
    mapper = RelationshipMapper()
    
    # Map relationships for each item
    mapped_items = []
    for item in knowledge_items:
        mapped = mapper.verify(item, existing_knowledge=knowledge_items)
        mapped_items.append(mapped)
    
    # Show a relationship graph
    print_relationship_graph(mapped_items)
    
    print_section("3. Confidence Scoring")
    
    print("Calculating confidence scores for knowledge items...")
    time.sleep(1)
    
    scorer = ConfidenceScorer()
    
    # Score a sample item
    scored_item = scorer.verify(knowledge_items[0])
    
    print("\nConfidence scoring result:")
    print_json(scored_item["metadata"])
    
    print("\nConfidence factors:")
    print_json(scored_item["metadata"]["confidence_factors"])
    
    # Return verified items
    return mapped_items

def demo_knowledge_integration(knowledge_items):
    """Demonstrate knowledge integration."""
    print_header("Knowledge Integration Demo")
    
    if not knowledge_items:
        print(f"{Fore.RED}No knowledge items to integrate. Run extraction demo first.{Style.RESET_ALL}")
        return
    
    print_section("1. Knowledge Formatting Options")
    
    # Select a sample item
    sample_item = knowledge_items[0]
    
    print(f"Sample knowledge item: {sample_item['statement']}")
    
    integrator = PromptKnowledgeIntegrator()
    
    # Show different format options
    format_types = ["default", "brief", "detailed", "contrastive", "rule", "example"]
    
    for format_type in format_types:
        formatted = integrator.format_knowledge(sample_item, format_type)
        print(f"\n{Fore.GREEN}{format_type.upper()} format:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{formatted}{Style.RESET_ALL}")
    
    print_section("2. Prompt Integration Strategies")
    
    # Create a sample prompt state
    sample_prompt_text = """
    Role: Medical Text Analyzer
    Task: Analyze the provided medical text to identify important medical entities.
    
    Steps:
    - Read the text carefully
    - Identify disease mentions
    - Identify drug mentions
    - Identify lab test results
    - Report all findings
    
    Output Format: List entities by category (Diseases, Drugs, Lab Tests).
    """
    
    sample_prompt = PromptState(sample_prompt_text)
    
    print("Original prompt:")
    print_prompt(sample_prompt)
    
    # Demonstrate different placement strategies
    placement_strategy = PlacementStrategy()
    format_strategy = FormatSelectionStrategy()
    
    placements = [
        "knowledge_section",
        "role_description",
        "task_description", 
        "step_instructions",
        "format_instructions",
        "constraints"
    ]
    
    # Select a suitable knowledge item for integration
    entity_item = next((item for item in knowledge_items 
                       if item.get("type") == "entity_classification"), knowledge_items[0])
    
    print("\nDemonstrating different placement strategies...")
    print(f"Knowledge to integrate: {Fore.YELLOW}{entity_item['statement']}{Style.RESET_ALL}")
    
    # Show automatic placement selection
    auto_placement = placement_strategy.select_placement(entity_item, sample_prompt)
    auto_format = format_strategy.select_format(entity_item, sample_prompt, auto_placement)
    
    print(f"\n{Fore.GREEN}Automatic strategy selection:{Style.RESET_ALL}")
    print(f"Recommended Placement: {Fore.CYAN}{auto_placement}{Style.RESET_ALL}")
    print(f"Recommended Format: {Fore.CYAN}{auto_format}{Style.RESET_ALL}")
    
    # Integrate with automatic strategy
    integrated_auto = integrator.integrate(sample_prompt, entity_item)
    
    print("\nIntegrated prompt (automatic strategy):")
    print_prompt(integrated_auto)
    
    # Select a different placement for comparison
    alt_placement = next(p for p in placements if p != auto_placement)
    integrated_alt = integrator.integrate(
        sample_prompt, 
        entity_item,
        override_placement=alt_placement
    )
    
    print(f"\nIntegrated prompt (alternative placement: {alt_placement}):")
    print_prompt(integrated_alt)
    
    print_section("3. Multiple Knowledge Integration with Conflict Resolution")
    
    # Select multiple items of different types
    items_to_integrate = []
    
    # Try to get one of each type
    types_added = set()
    for item in knowledge_items:
        item_type = item.get("type", "")
        if item_type and item_type not in types_added:
            items_to_integrate.append(item)
            types_added.add(item_type)
            if len(items_to_integrate) >= 3:
                break
    
    # Fall back to any items if not enough variety
    if len(items_to_integrate) < 3:
        remaining = [item for item in knowledge_items if item not in items_to_integrate]
        items_to_integrate.extend(remaining[:3 - len(items_to_integrate)])
    
    # Add a conflicting item
    if items_to_integrate:
        conflict_item = items_to_integrate[0].copy()
        conflict_item["id"] = "k_conflict"
        conflict_item["statement"] = "Contradictory information: " + conflict_item["statement"]
        items_to_integrate.append(conflict_item)
    
    print(f"Integrating {len(items_to_integrate)} knowledge items, including potential conflicts...")
    time.sleep(1)
    
    conflict_resolver = ConflictResolutionStrategy()
    
    # Extract existing knowledge from the prompt
    existing = integrator._extract_existing_knowledge(sample_prompt.text)
    
    # Resolve conflicts
    resolved_items = conflict_resolver.resolve_conflicts(items_to_integrate, existing)
    
    print(f"\nAfter conflict resolution: {len(resolved_items)} items remain for integration")
    
    # Integrate all resolved items
    final_prompt = sample_prompt
    for item in resolved_items:
        final_prompt = integrator.integrate(final_prompt, item)
    
    print("\nFinal integrated prompt:")
    print_prompt(final_prompt)

def demo_knowledge_base():
    """Demonstrate knowledge base functionality."""
    print_header("Knowledge Base Demo")
    
    # Create an in-memory knowledge base for demo
    import tempfile
    temp_dir = tempfile.gettempdir()
    kb_dir = Path(temp_dir) / "demo_kb"
    kb_dir.mkdir(exist_ok=True)
    
    domain_dir = kb_dir / "domain_knowledge"
    error_dir = kb_dir / "error_patterns"
    
    domain_dir.mkdir(exist_ok=True)
    error_dir.mkdir(exist_ok=True)
    
    print(f"Created temporary knowledge base at {kb_dir}")
    
    kb = KnowledgeBase(domain_dir, error_dir)
    
    print_section("1. Adding Knowledge to Knowledge Base")
    
    # Create sample knowledge items
    sample_items = [
        {
            "id": "k_med_001",
            "type": "entity_classification",
            "statement": "HbA1c is a lab test that measures average blood glucose levels, not a disease.",
            "entities": ["HbA1c"],
            "relations": [
                {"subject": "HbA1c", "predicate": "isA", "object": "lab_test"}
            ],
            "metadata": {
                "source": "error_feedback",
                "domain": "medical",
                "confidence": 0.85
            }
        },
        {
            "id": "k_med_002",
            "type": "procedural_knowledge",
            "statement": "Process for calculating GFR",
            "procedure_topic": "GFR calculation",
            "procedure_steps": [
                "Gather patient information (age, sex, race, and serum creatinine level)",
                "Choose the appropriate GFR estimation equation",
                "Input the values into the selected equation",
                "Calculate the estimated GFR",
                "Interpret the result according to clinical guidelines"
            ],
            "metadata": {
                "source": "text_extraction",
                "domain": "medical",
                "confidence": 0.9
            }
        },
        {
            "id": "k_med_003",
            "type": "conceptual_knowledge",
            "statement": "ACE inhibitors are a class of medication used primarily for hypertension and heart failure.",
            "entities": ["ACE inhibitors"],
            "relations": [
                {"subject": "ACE inhibitors", "predicate": "isA", "object": "medication_class"},
                {"subject": "ACE inhibitors", "predicate": "treatsCondition", "object": "hypertension"},
                {"subject": "ACE inhibitors", "predicate": "treatsCondition", "object": "heart failure"}
            ],
            "metadata": {
                "source": "text_extraction",
                "domain": "medical",
                "confidence": 0.95
            }
        }
    ]
    
    # Add items to knowledge base
    for item in sample_items:
        print(f"Adding knowledge item: {item['id']} - {item['statement'][:50]}...")
        kb.add_knowledge(item)
    
    print(f"\nAdded {len(sample_items)} items to knowledge base")
    
    print_section("2. Retrieving Knowledge from Knowledge Base")
    
    # Get domain knowledge
    medical_knowledge = kb.get_domain_knowledge("medical")
    
    print(f"Retrieved {len(medical_knowledge)} items from 'medical' domain")
    
    # Get a specific knowledge item
    item_id = "k_med_003"
    item = kb.get_knowledge(item_id)
    
    print(f"\nRetrieved knowledge item {item_id}:")
    print_json(item)
    
    print_section("3. Searching and Querying Knowledge")
    
    print("Search by text query:")
    results1 = kb.search_knowledge("ACE inhibitors")
    
    print(f"Found {len(results1)} results for 'ACE inhibitors':")
    for result in results1:
        print(f"- {result['id']}: {result['statement'][:50]}...")
    
    print("\nSearch by entities:")
    results2 = kb.query_by_entities(["HbA1c"])
    
    print(f"Found {len(results2)} results for entity 'HbA1c':")
    for result in results2:
        print(f"- {result['id']}: {result['statement'][:50]}...")
    
    print_section("4. Knowledge Base Statistics")
    
    # Get stats
    stats = kb.get_knowledge_stats()
    
    print("Knowledge Base Statistics:")
    print(f"Total items: {stats['total_items']}")
    print(f"Domains: {stats['domains']}")
    print("Types:")
    for k_type, count in stats.get("types", {}).items():
        print(f"  - {k_type}: {count} items")

def demo_domain_manager():
    """Demonstrate domain knowledge manager functionality."""
    print_header("Domain Knowledge Manager Demo")
    
    # Create a domain knowledge manager
    manager = DomainKnowledgeManager()
    
    print_section("1. Integrated Extraction and Verification Workflow")
    
    # Sample errors data
    errors = [
        {
            "example_id": "e3",
            "example": {
                "text": "The study included patients with BMI > 30 kg/m².",
                "expected": "condition_mention"
            },
            "error_type": "entity_confusion",
            "actual": "measurement",
            "description": "Failed to recognize that BMI > 30 indicates obesity."
        },
        {
            "example_id": "e4",
            "example": {
                "text": "The patient presented with ST elevation in leads V1-V4.",
                "expected": "condition_mention"
            },
            "error_type": "entity_confusion",
            "actual": "test_result",
            "description": "Failed to recognize that ST elevation suggests myocardial infarction."
        }
    ]
    
    print("Integrated workflow: Extract → Verify → Store")
    time.sleep(1)
    
    print("\nStep 1: Extract knowledge from errors")
    extracted = manager.extract_knowledge(errors, extractor_type="error", domain="medical")
    
    print(f"Extracted {len(extracted)} knowledge items")
    for item in extracted:
        print(f"- {item['statement']}")
    
    print("\nStep 2: Verify knowledge")
    verified = manager.verify_knowledge(extracted, verify_types=["consistency", "confidence"])
    
    print(f"Verified {len(verified)} knowledge items")
    for item in verified:
        confidence = item["metadata"].get("confidence", 0)
        print(f"- {item['statement']} (Confidence: {confidence:.2f})")
    
    print("\nStep 3: Extract relationships between knowledge items")
    related = manager.verify_knowledge(verified, verify_types=["relationship"])
    
    print("Added relationship metadata")
    for item in related:
        relationships = item["metadata"].get("relationship_mapping", {}).get("relationships", [])
        print(f"- {item['statement']} ({len(relationships)} relationships)")
    
    # Don't actually add to knowledge base in demo
    # print("\nStep 4: Add to knowledge base")
    # added_ids = manager.add_knowledge(related, verify=False, extract_relationships=False)
    # print(f"Added {len(added_ids)} items to knowledge base")
    
    print_section("2. From Error to Prompt Enhancement")
    
    # Create a sample medical prompt
    medical_prompt_text = """
    Role: Medical Text Analyzer
    Task: Analyze the provided clinical notes to identify medical conditions and diagnoses.
    
    Steps:
    - Read the clinical notes carefully
    - Identify all mentions of medical conditions
    - Report conditions with their supporting evidence from the text
    
    Output Format: List each condition with the relevant text evidence.
    """
    
    medical_prompt = PromptState(medical_prompt_text)
    
    print("Original prompt:")
    print_prompt(medical_prompt)
    
    # Create integrator
    integrator = PromptKnowledgeIntegrator()
    
    # Integrate the knowledge from errors
    enhanced_prompt = medical_prompt
    for item in related:
        enhanced_prompt = integrator.integrate(enhanced_prompt, item)
    
    print("\nEnhanced prompt with error-derived knowledge:")
    print_prompt(enhanced_prompt)
    
    print_section("3. Domain-Specific Knowledge Application")
    
    print(f"{Fore.CYAN}In a real application, knowledge from each domain would be carefully curated,\nverified by experts, and integrated strategically into prompts to guide the LLM.{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}This enables the system to automatically enhance prompts with domain expertise,\nmaking them more effective and accurate.{Style.RESET_ALL}")
    
    # Return enhanced prompt as demo result
    return enhanced_prompt

def main():
    """Main function"""
    print_header("Domain Knowledge Integration Demo")
    
    menu_options = [
        "Knowledge Extraction Demo",
        "Knowledge Verification Demo",
        "Knowledge Integration Demo",
        "Knowledge Base Demo",
        "Domain Knowledge Manager Demo",
        "Complete Process Demo",
        "Exit"
    ]
    
    # Store demo results for reuse
    results = {
        "extracted_items": [],
        "verified_items": []
    }
    
    while True:
        print_section("Function Options")
        for i, option in enumerate(menu_options, 1):
            print(f"{i}. {option}")
        
        try:
            choice = input(f"\n{Fore.CYAN}Please select a function (1-{len(menu_options)}): {Style.RESET_ALL}")
            choice = int(choice)
            
            if choice == 1:
                results["extracted_items"] = demo_knowledge_extraction()
            elif choice == 2:
                results["verified_items"] = demo_knowledge_verification(results["extracted_items"])
            elif choice == 3:
                demo_knowledge_integration(results["verified_items"] or results["extracted_items"])
            elif choice == 4:
                demo_knowledge_base()
            elif choice == 5:
                demo_domain_manager()
            elif choice == 6:
                # Complete process demo
                print_header("Complete Domain Knowledge Integration Process")
                extracted = demo_knowledge_extraction()
                verified = demo_knowledge_verification(extracted)
                demo_knowledge_integration(verified)
                demo_knowledge_base()
                demo_domain_manager()
            elif choice == 7:
                print("\nThank you for using the Domain Knowledge Integration Demo!\n")
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