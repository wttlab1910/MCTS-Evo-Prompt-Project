"""
API routes for prompt processing.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.api.models.request_models import InputTextRequest, PromptExpansionRequest, ModelTrainingRequest
from app.api.models.response_models import ProcessedInputResponse, ExpandedPromptResponse, TrainingStatusResponse
from app.services.prompt_service import PromptService
from app.api.middleware.auth import verify_api_key
from app.utils.logger import get_logger

logger = get_logger("api.prompt_routes")

# Create router
router = APIRouter()

# Create service instance
prompt_service = PromptService()

@router.post("/process", response_model=ProcessedInputResponse, dependencies=[Depends(verify_api_key)])
async def process_input(request: InputTextRequest):
    """
    Process input text to separate prompt and data, analyze task, and expand prompt.
    
    Args:
        request: Input text request.
        
    Returns:
        Processed input response.
    """
    try:
        result = prompt_service.process_input(request.text)
        return ProcessedInputResponse(
            prompt=result["prompt"],
            data=result["data"],
            task_type=result["task_analysis"]["task_type"],
            category=result["task_analysis"]["category"],
            expanded_prompt=result["expanded_prompt"]
        )
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

@router.post("/expand", response_model=ExpandedPromptResponse, dependencies=[Depends(verify_api_key)])
async def expand_prompt(request: PromptExpansionRequest):
    """
    Expand a prompt with structured enhancements.
    
    Args:
        request: Prompt expansion request.
        
    Returns:
        Expanded prompt response.
    """
    try:
        expanded_prompt = prompt_service.expand_prompt(request.prompt, request.task_type)
        return ExpandedPromptResponse(
            original_prompt=request.prompt,
            expanded_prompt=expanded_prompt
        )
    except Exception as e:
        logger.error(f"Error expanding prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error expanding prompt: {str(e)}")

@router.post("/train", response_model=TrainingStatusResponse, dependencies=[Depends(verify_api_key)])
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """
    Train the prompt expansion model.
    
    Args:
        request: Model training request.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Training status response.
    """
    try:
        # Start training in background
        background_tasks.add_task(prompt_service.train_expansion_model, request.epochs, request.batch_size)
        
        return TrainingStatusResponse(
            status="started",
            message="Model training started in background"
        )
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting model training: {str(e)}")