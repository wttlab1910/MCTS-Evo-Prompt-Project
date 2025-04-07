"""
API response models.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class PromptResponse(BaseModel):
    """
    Response model for prompt operations.
    """
    
    original: str = Field(
        ...,
        description="Original input text"
    )
    
    prompt: str = Field(
        ...,
        description="Extracted prompt part"
    )
    
    data: str = Field(
        ...,
        description="Extracted data part"
    )
    
    expanded: str = Field(
        ...,
        description="Expanded prompt"
    )

class PromptAnalysisResponse(BaseModel):
    """
    Response model for prompt analysis.
    """
    
    original_input: str = Field(
        ...,
        description="Original input text"
    )
    
    prompt: str = Field(
        ...,
        description="Extracted prompt part"
    )
    
    data: str = Field(
        ...,
        description="Extracted data part"
    )
    
    task_analysis: Dict[str, Any] = Field(
        ...,
        description="Task analysis results"
    )
    
    expanded_prompt: str = Field(
        ...,
        description="Expanded prompt"
    )

class OptimizationResponse(BaseModel):
    """
    Response model for optimization operations.
    """
    
    optimization_id: str = Field(
        ...,
        description="Optimization job ID"
    )
    
    status: str = Field(
        ...,
        description="Optimization status",
        example="running"
    )
    
    message: str = Field(
        ...,
        description="Status message",
        example="Optimization in progress"
    )
    
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Optimization result"
    )
    
    progress: Optional[float] = Field(
        None,
        description="Optimization progress (0.0 to 1.0)"
    )
    
    stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Optimization statistics"
    )

class KnowledgeEntryResponse(BaseModel):
    """
    Response model for knowledge entry operations.
    """
    
    id: str = Field(
        ...,
        description="Knowledge entry ID"
    )
    
    knowledge_type: str = Field(
        ...,
        description="Type of knowledge entry"
    )
    
    statement: str = Field(
        ...,
        description="Knowledge statement"
    )
    
    domain: str = Field(
        ...,
        description="Knowledge domain"
    )
    
    metadata: Dict[str, Any] = Field(
        ...,
        description="Additional metadata"
    )
    
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )

class KnowledgeListResponse(BaseModel):
    """
    Response model for knowledge entry list operations.
    """
    
    entries: List[KnowledgeEntryResponse] = Field(
        ...,
        description="List of knowledge entries"
    )
    
    count: int = Field(
        ...,
        description="Total number of entries"
    )