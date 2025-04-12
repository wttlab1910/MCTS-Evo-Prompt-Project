"""
Task-specific action generator module.

This module provides specialized action generators for different task types.
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import List, Dict, Any, Optional, Callable
from app.core.mdp.state import PromptState
from app.core.mdp.action import Action, create_action
from app.utils.logger import get_logger

logger = get_logger("actions.task_actions")


def get_task_action_generator(task_name: str) -> Optional[Callable[[PromptState], List[Action]]]:
    """
    Get a task-specific action generator.
    
    Args:
        task_name: Name of the task.
        
    Returns:
        Action generator function or None if no specialized generator exists.
    """
    task_generators = {
        # BigBench tasks
        "penguins_in_a_table": generate_table_task_actions,
        "object_counting": generate_counting_task_actions,
        "temporal_sequences": generate_sequence_task_actions,
        "causal_judgment": generate_causal_task_actions,
        "epistemic": generate_epistemic_task_actions,
        "geometric_shapes": generate_geometric_task_actions,
        
        # NLP tasks
        "ncbi": generate_ncbi_task_actions,
        "trec": generate_trec_task_actions,
        "biosses": generate_biosses_task_actions,
        "cb": generate_cb_task_actions,
        "med_qa": generate_med_qa_task_actions,
        "subj": generate_subj_task_actions
    }
    
    logger.info(f"Requested actions for task: {task_name}")
    
    # Handle case normalization and format variations
    normalized_task_name = task_name.lower().replace('-', '_').replace(' ', '_')
    
    if normalized_task_name in task_generators:
        logger.info(f"Found task-specific action generator for {normalized_task_name}")
        return task_generators[normalized_task_name]
    
    # Default generic actions if no specific generator exists
    logger.info(f"No specific action generator for {task_name}, using generic actions")
    return generate_generic_task_actions

def generate_table_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for table-based tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Table structure understanding actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "When analyzing tables, first identify all columns and their data types",
        "domain": "table_structure"
    }))
    
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "For tables with penguins data, look for species, measurements, location, and count columns",
        "domain": "penguin_data"
    }))
    
    # Query handling actions
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "For tabular data, questions typically fall into four categories: lookup (direct retrieval), counting (how many items meet criteria), comparison (finding max/min), or calculation (averages, sums, etc.)",
        "target": "task"
    }))
    
    # Step-based actions for table processing
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "First, identify the structure of the table (columns and their meanings)",
            "Determine what type of question is being asked (lookup, counting, comparison, or calculation)",
            "Extract the relevant data from the table",
            "Apply the appropriate operation to answer the question",
            "Verify your answer by checking against the table"
        ]
    }))
    
    # Format actions for table responses
    actions.append(create_action("specify_format", parameters={
        "format_text": "For table questions, start with a brief description of how you'll approach the question, followed by the exact answer extracted or calculated from the table."
    }))
    
    actions.append(create_action("add_constraint", parameters={
        "constraint_text": "When working with table data, always verify counts, sums, and averages before giving a final answer."
    }))
    
    # Example-based actions
    actions.append(create_action("add_example", parameters={
        "example_text": "Example: Given a penguin table with columns [Species, Island, Bill Length, Flipper Length], if asked 'What is the average bill length?', first identify the Bill Length column, extract all values, then calculate and report the average.",
        "example_type": "table_calculation"
    }))
    
    return actions

def generate_counting_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for counting tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Counting methodology actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "For counting problems, first identify all items that match the counting criteria",
        "domain": "counting_methodology"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Identify what needs to be counted",
            "Establish counting criteria (what qualifies to be counted)",
            "Count systematically to avoid missing items or counting twice",
            "Verify the count by rechecking",
            "Report the final count"
        ]
    }))
    
    # Specific counting strategies
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "For object counting, use grouping strategies when helpful: count by category, location, or other logical groups",
        "target": "task"
    }))
    
    actions.append(create_action("add_constraint", parameters={
        "constraint_text": "Always double-check your count, especially with complex arrangements or multiple object types."
    }))
    
    return actions

def generate_sequence_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for sequence-based tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Sequence pattern actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "For sequence problems, look for arithmetic patterns (adding, multiplying), geometric patterns, or repeating sequences",
        "domain": "sequence_patterns"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Identify the given sequence elements",
            "Analyze the pattern (arithmetic, geometric, repeating, etc.)",
            "Determine the rule that generates the sequence",
            "Apply the rule to find the next element(s)",
            "Verify by checking that the rule applies to all given elements"
        ]
    }))
    
    # Temporal sequence specific actions
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "For temporal sequences, consider cyclic patterns (daily, weekly, monthly, seasonal) and progression patterns",
        "target": "task"
    }))
    
    return actions

def generate_causal_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for causal judgment tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Causal reasoning actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "Causal judgment requires distinguishing correlation from causation. Consider alternative explanations and confounding factors.",
        "domain": "causal_reasoning"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Identify the proposed cause and effect",
            "Examine if a mechanism connects cause and effect",
            "Consider alternative explanations (confounders, coincidence)",
            "Evaluate strength of evidence for causation",
            "Make a judgement about causal relationship"
        ]
    }))
    
    actions.append(create_action("add_constraint", parameters={
        "constraint_text": "When making causal judgments, be careful not to confuse correlation with causation. Two events happening together doesn't necessarily mean one caused the other."
    }))
    
    return actions

def generate_epistemic_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for epistemic reasoning tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Epistemic reasoning actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "Epistemic reasoning involves analyzing beliefs, knowledge, and certainty. Consider what is known vs. inferred vs. assumed.",
        "domain": "epistemic_reasoning"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Identify what information is explicitly known",
            "Distinguish between facts, inferences, and assumptions",
            "Determine what can be logically deduced from known information",
            "Assess confidence levels for conclusions",
            "Present final judgment with appropriate certainty"
        ]
    }))
    
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "For epistemic tasks, carefully track the source and reliability of different pieces of information",
        "target": "task"
    }))
    
    return actions

def generate_geometric_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for geometric reasoning tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Geometric reasoning actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "For geometric problems, consider properties like shape, size, position, orientation, and transformations",
        "domain": "geometric_reasoning"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Identify all geometric shapes and their properties",
            "Note spatial relationships between objects",
            "Apply relevant geometric principles or formulas",
            "Consider transformations (rotation, reflection, translation)",
            "Verify the solution using geometric properties"
        ]
    }))
    
    actions.append(create_action("add_constraint", parameters={
        "constraint_text": "When solving geometric problems, draw out the shapes if helpful, and carefully track how shapes relate to each other."
    }))
    
    return actions

# New task action generators

def generate_ncbi_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for NCBI disease entity recognition tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # NER methodology actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "Disease Named Entity Recognition requires identifying all mentions of diseases, disorders, and medical conditions in text.",
        "domain": "biomedical_ner"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Read through the entire text to understand context",
            "Identify all potential disease mentions",
            "Normalize disease terms (e.g., 'Alzheimer's' and 'Alzheimer disease' refer to the same entity)",
            "Check for related terms and abbreviations",
            "Return the complete set of unique disease entities"
        ]
    }))
    
    actions.append(create_action("specify_format", parameters={
        "format_text": "Return all disease entities as a comma-separated list inside curly braces, like this: {entity1, entity2, entity3}. If no disease entities are present, return an empty list: {}."
    }))
    
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "Pay attention to disease mentions that span multiple words or contain descriptive modifiers.",
        "target": "task"
    }))
    
    return actions

def generate_trec_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for TREC question classification tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Question classification actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "TREC question classification assigns questions to categories based on the type of information being sought: abbreviation, entity, description, human, location, or numeric value.",
        "domain": "question_classification"
    }))
    
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "Category details: (A) Abbreviation - seeking expansion of acronyms, (B) Entity - questions about things, (C) Description - seeking explanations, (D) Human - questions about people, (E) Location - geographical questions, (F) Numeric - questions seeking numbers.",
        "domain": "trec_categories"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Identify question words (who, what, when, where, why, how)",
            "Determine what type of answer is being sought",
            "Match to the appropriate TREC category (A-F)",
            "Consider any ambiguities between categories",
            "Select the most appropriate category"
        ]
    }))
    
    actions.append(create_action("add_example", parameters={
        "example_text": "Examples: 'What is the capital of France?' → E (Location), 'Who invented the telephone?' → D (Human), 'When was the Declaration of Independence signed?' → F (Numeric value)",
        "example_type": "question_classification"
    }))
    
    return actions

def generate_biosses_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for BIOSSES sentence similarity tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Sentence similarity actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "BIOSSES evaluates semantic similarity between biomedical sentences on a scale from 'not similar' to 'somewhat similar' to 'similar'.",
        "domain": "sentence_similarity"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Compare the key biomedical concepts in both sentences",
            "Identify shared terminology and synonymous terms",
            "Consider the biological processes or relationships described",
            "Evaluate overall semantic meaning",
            "Assign a similarity category (not similar, somewhat similar, or similar)"
        ]
    }))
    
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "Focus on biomedical meaning rather than surface text similarity. Consider domain-specific synonyms and related concepts.",
        "target": "task"
    }))
    
    return actions

def generate_cb_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for CommitmentBank (CB) entailment tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Entailment reasoning actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "The CommitmentBank task involves determining the relationship between a premise and hypothesis: entailment (premise implies hypothesis), contradiction (premise implies hypothesis is false), or neutral (premise neither implies hypothesis nor its negation).",
        "domain": "textual_entailment"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Carefully read the premise and hypothesis",
            "Identify key information and claims in the premise",
            "Determine if the hypothesis must be true given the premise (entailment)",
            "Determine if the hypothesis must be false given the premise (contradiction)",
            "If neither, the relationship is neutral"
        ]
    }))
    
    actions.append(create_action("add_constraint", parameters={
        "constraint_text": "Be cautious about commonsense assumptions. Focus strictly on what the premise logically entails."
    }))
    
    actions.append(create_action("add_example", parameters={
        "example_text": "Example: Premise: 'All cats are mammals.' Hypothesis: 'Some mammals are cats.' → Entailment (must be true given the premise)",
        "example_type": "entailment"
    }))
    
    return actions

def generate_med_qa_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for medical QA tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Medical QA actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "Medical question answering requires understanding medical terminology, conditions, treatments, and their relationships.",
        "domain": "medical_knowledge"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Read the question carefully, noting medical terms",
            "Examine all answer options thoroughly",
            "Apply medical knowledge to evaluate each option",
            "Eliminate clearly incorrect options",
            "Select the most accurate answer option"
        ]
    }))
    
    actions.append(create_action("add_constraint", parameters={
        "constraint_text": "Medical questions often require precise understanding of terminology. Pay close attention to specific medical terms and their exact meanings."
    }))
    
    actions.append(create_action("specify_format", parameters={
        "format_text": "Indicate your final answer clearly by specifying the chosen option letter (e.g., 'The answer is A')."
    }))
    
    return actions

def generate_subj_task_actions(state: PromptState) -> List[Action]:
    """
    Generate actions for subjectivity classification tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Subjectivity classification actions
    actions.append(create_action("add_domain_knowledge", parameters={
        "knowledge_text": "Subjectivity classification distinguishes between objective statements (facts, verifiable information) and subjective statements (opinions, judgments, personal views).",
        "domain": "subjectivity_analysis"
    }))
    
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Read the text and identify factual vs. opinion-based content",
            "Look for objective markers (verifiable facts, measurements, dates)",
            "Look for subjective markers (evaluative language, personal judgments, emotions)",
            "Weigh the overall character of the text",
            "Classify as either 'Objective' or 'Subjective'"
        ]
    }))
    
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "Subjective text typically contains opinion words (good, bad, best, worst), emotion words, and personal judgments. Objective text focuses on verifiable facts without evaluation.",
        "target": "task"
    }))
    
    actions.append(create_action("add_example", parameters={
        "example_text": "Example: 'The temperature today is 75 degrees Fahrenheit.' → Objective. 'The weather today is wonderful.' → Subjective.",
        "example_type": "subjectivity"
    }))
    
    return actions

def generate_generic_task_actions(state: PromptState) -> List[Action]:
    """
    Generate generic actions that can apply to most tasks.
    
    Args:
        state: Current prompt state.
        
    Returns:
        List of applicable actions.
    """
    actions = []
    
    # Generic workflow actions
    actions.append(create_action("modify_workflow", parameters={
        "steps": [
            "Carefully read and understand the question",
            "Identify the key concepts and requirements",
            "Apply relevant knowledge and reasoning",
            "Provide a clear, concise answer",
            "Verify that the answer directly addresses the question"
        ]
    }))
    
    # General improvement actions
    actions.append(create_action("add_constraint", parameters={
        "constraint_text": "Always provide step-by-step reasoning before giving your final answer."
    }))
    
    actions.append(create_action("add_explanation", parameters={
        "explanation_text": "Break down complex problems into smaller, manageable parts before solving them.",
        "target": "task"
    }))
    
    # Format guidance
    actions.append(create_action("specify_format", parameters={
        "format_text": "Start with a brief analysis of the question, then provide your reasoning process, and conclude with a clear final answer."
    }))
    
    return actions