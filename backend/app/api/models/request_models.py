"""
API request models.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    """
    Request model for prompt operations.
    """
    
    input_text: str = Field(
        ...,
        description="Input text containing prompt and optionally data",
        example="Classify the sentiment of this review. This movie was amazing!"
    )

class EvaluatePromptRequest(PromptRequest):
    """
    Request model for prompt evaluation.
    """
    
    expected_output: Optional[str] = Field(
        None,
        description="Expected output for evaluation",
        example="Positive"
    )
    
    use_data: bool = Field(
        True,
        description="Whether to use the data part of the input text for evaluation"
    )

class OptimizationRequest(PromptRequest):
    """
    Request model for prompt optimization.
    """
    
    expected_output: Optional[str] = Field(
        None,
        description="Expected output for optimization",
        example="Positive"
    )
    
    iterations: int = Field(
        50,
        description="Number of optimization iterations",
        ge=1,
        le=1000
    )
    
    timeout: int = Field(
        300,
        description="Optimization timeout in seconds",
        ge=10,
        le=3600
    )
    
    validation_examples: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Validation examples for optimization",
        example=[
            {"input": "This movie was amazing!", "output": "Positive"},
            {"input": "I hated this product.", "output": "Negative"}
        ]
    )

class KnowledgeEntryRequest(BaseModel):
    """
    Request model for knowledge entry operations.
    """
    
    knowledge_type: str = Field(
        ...,
        description="Type of knowledge entry",
        example="entity_classification"
    )
    
    statement: str = Field(
        ...,
        description="Knowledge statement",
        example="PAH is a gene, not a disease"
    )
    
    domain: str = Field(
        ...,
        description="Knowledge domain",
        example="biomedical"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
        example={"source": "error_feedback", "confidence": 0.95}
    )