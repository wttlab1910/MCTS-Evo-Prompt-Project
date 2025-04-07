"""
Prompt-related API endpoints.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from app.api.models.request_models import PromptRequest, EvaluatePromptRequest
from app.api.models.response_models import PromptResponse, PromptAnalysisResponse
from app.services.prompt_service import PromptService
from app.utils.logger import get_logger

logger = get_logger("api.endpoints.prompt_routes")

router = APIRouter()

def get_prompt_service() -> PromptService:
    """
    Get prompt service instance.
    
    Returns:
        PromptService instance.
    """
    return PromptService()

@router.post("/analyze", response_model=PromptAnalysisResponse)
async def analyze_prompt(
    request: PromptRequest = Body(...),
    prompt_service: PromptService = Depends(get_prompt_service)
):
    """
    Analyze a prompt to understand its task and structure.
    
    Args:
        request: Prompt request.
        prompt_service: Prompt service.
        
    Returns:
        Prompt analysis response.
    """
    try:
        result = prompt_service.process_input(request.input_text)
        
        return PromptAnalysisResponse(
            original_input=request.input_text,
            prompt=result["prompt"],
            data=result["data"],
            task_analysis=result["task_analysis"],
            expanded_prompt=result["expanded_prompt"]
        )
    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze prompt: {str(e)}")

@router.post("/expand", response_model=PromptResponse)
async def expand_prompt(
    request: PromptRequest = Body(...),
    prompt_service: PromptService = Depends(get_prompt_service)
):
    """
    Expand a prompt using prompt engineering techniques.
    
    Args:
        request: Prompt request.
        prompt_service: Prompt service.
        
    Returns:
        Expanded prompt response.
    """
    try:
        # Process input to extract prompt and data
        result = prompt_service.process_input(request.input_text)
        
        # Expand the prompt
        expanded_prompt = prompt_service.expand_prompt(
            result["prompt"], 
            result["task_analysis"].get("task_type")
        )
        
        return PromptResponse(
            original=request.input_text,
            prompt=result["prompt"],
            data=result["data"],
            expanded=expanded_prompt
        )
    except Exception as e:
        logger.error(f"Error expanding prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to expand prompt: {str(e)}")

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_prompt(
    request: EvaluatePromptRequest = Body(...),
    prompt_service: PromptService = Depends(get_prompt_service)
):
    """
    Evaluate a prompt's quality.
    
    Args:
        request: Evaluate prompt request.
        prompt_service: Prompt service.
        
    Returns:
        Evaluation results.
    """
    try:
        # Process input to extract prompt and data
        result = prompt_service.process_input(request.input_text)
        
        # Evaluate the prompt
        evaluation = prompt_service.evaluate_prompt(
            result["prompt"],
            result["task_analysis"].get("task_type"),
            result["data"] if request.use_data else None,
            expected_output=request.expected_output
        )
        
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to evaluate prompt: {str(e)}")