"""
Optimization-related API endpoints.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from app.api.models.request_models import OptimizationRequest
from app.api.models.response_models import OptimizationResponse
from app.services.optimization_service import OptimizationService
from app.utils.logger import get_logger

logger = get_logger("api.endpoints.optimization_routes")

router = APIRouter()

def get_optimization_service() -> OptimizationService:
    """
    Get optimization service instance.
    
    Returns:
        OptimizationService instance.
    """
    return OptimizationService()

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_prompt(
    request: OptimizationRequest = Body(...),
    optimization_service: OptimizationService = Depends(get_optimization_service)
):
    """
    Optimize a prompt using MCTS with evolutionary algorithms.
    
    Args:
        request: Optimization request.
        optimization_service: Optimization service.
        
    Returns:
        Optimization response.
    """
    try:
        # Start optimization
        optimization_id = await optimization_service.start_optimization(
            input_text=request.input_text,
            expected_output=request.expected_output,
            iterations=request.iterations,
            timeout=request.timeout,
            validation_examples=request.validation_examples
        )
        
        # This is a long-running operation, so we return the optimization ID
        # The client will need to poll for results
        return OptimizationResponse(
            optimization_id=optimization_id,
            status="running",
            message="Optimization started. Use /status/{optimization_id} to check progress."
        )
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

@router.get("/status/{optimization_id}", response_model=OptimizationResponse)
async def get_optimization_status(
    optimization_id: str,
    optimization_service: OptimizationService = Depends(get_optimization_service)
):
    """
    Get the status of an optimization job.
    
    Args:
        optimization_id: Optimization ID.
        optimization_service: Optimization service.
        
    Returns:
        Optimization status response.
    """
    try:
        # Get optimization status
        status = await optimization_service.get_optimization_status(optimization_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Optimization job {optimization_id} not found")
        
        return OptimizationResponse(
            optimization_id=optimization_id,
            status=status["status"],
            message=status["message"],
            result=status.get("result"),
            progress=status.get("progress"),
            stats=status.get("stats")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization status: {str(e)}")

@router.delete("/cancel/{optimization_id}", response_model=Dict[str, Any])
async def cancel_optimization(
    optimization_id: str,
    optimization_service: OptimizationService = Depends(get_optimization_service)
):
    """
    Cancel an ongoing optimization job.
    
    Args:
        optimization_id: Optimization ID.
        optimization_service: Optimization service.
        
    Returns:
        Cancellation result.
    """
    try:
        # Cancel optimization
        success = await optimization_service.cancel_optimization(optimization_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Optimization job {optimization_id} not found or already completed")
        
        return {
            "optimization_id": optimization_id,
            "status": "cancelled",
            "message": "Optimization job cancelled successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel optimization: {str(e)}")