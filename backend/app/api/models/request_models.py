"""
Request models for API endpoints.
"""
from typing import Optional
from pydantic import BaseModel, Field

class InputTextRequest(BaseModel):
    """
    Request model for processing input text.
    """
    
    text: str = Field(..., description="Input text to process", min_length=1)

class PromptExpansionRequest(BaseModel):
    """
    Request model for expanding a prompt.
    """
    
    prompt: str = Field(..., description="Prompt to expand", min_length=1)
    task_type: Optional[str] = Field(None, description="Task type override")

class ModelTrainingRequest(BaseModel):
    """
    Request model for training the prompt expansion model.
    """
    
    epochs: int = Field(3, description="Number of training epochs", ge=1, le=10)
    batch_size: int = Field(8, description="Training batch size", ge=1, le=32)