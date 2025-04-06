"""
Response models for API endpoints.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ProcessedInputResponse(BaseModel):
    """
    Response model for processed input.
    """
    
    prompt: str = Field(..., description="Extracted prompt")
    data: str = Field(..., description="Extracted data")
    task_type: str = Field(..., description="Identified task type")
    category: str = Field(..., description="Task category")
    expanded_prompt: str = Field(..., description="Expanded prompt")

class ExpandedPromptResponse(BaseModel):
    """
    Response model for expanded prompt.
    """
    
    original_prompt: str = Field(..., description="Original prompt")
    expanded_prompt: str = Field(..., description="Expanded prompt")

class TrainingStatusResponse(BaseModel):
    """
    Response model for model training status.
    """
    
    status: str = Field(..., description="Status of the training process")
    message: str = Field(..., description="Status message")