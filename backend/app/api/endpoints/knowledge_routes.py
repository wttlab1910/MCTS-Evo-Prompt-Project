"""
Knowledge-related API endpoints.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from app.api.models.request_models import KnowledgeEntryRequest
from app.api.models.response_models import KnowledgeEntryResponse, KnowledgeListResponse
from app.services.knowledge_service import KnowledgeService
from app.utils.logger import get_logger

logger = get_logger("api.endpoints.knowledge_routes")

router = APIRouter()

def get_knowledge_service() -> KnowledgeService:
    """
    Get knowledge service instance.
    
    Returns:
        KnowledgeService instance.
    """
    return KnowledgeService()

@router.get("/entries", response_model=KnowledgeListResponse)
async def list_knowledge_entries(
    domain: Optional[str] = Query(None),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    List knowledge entries, optionally filtered by domain.
    
    Args:
        domain: Domain filter (optional).
        knowledge_service: Knowledge service.
        
    Returns:
        List of knowledge entries.
    """
    try:
        entries = await knowledge_service.list_entries(domain)
        
        return KnowledgeListResponse(
            entries=entries,
            count=len(entries)
        )
    except Exception as e:
        logger.error(f"Error listing knowledge entries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list knowledge entries: {str(e)}")

@router.get("/entries/{entry_id}", response_model=KnowledgeEntryResponse)
async def get_knowledge_entry(
    entry_id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Get a specific knowledge entry.
    
    Args:
        entry_id: Knowledge entry ID.
        knowledge_service: Knowledge service.
        
    Returns:
        Knowledge entry.
    """
    try:
        entry = await knowledge_service.get_entry(entry_id)
        
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Knowledge entry {entry_id} not found")
        
        return KnowledgeEntryResponse(**entry)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge entry: {str(e)}")

@router.post("/entries", response_model=KnowledgeEntryResponse)
async def create_knowledge_entry(
    request: KnowledgeEntryRequest = Body(...),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Create a new knowledge entry.
    
    Args:
        request: Knowledge entry request.
        knowledge_service: Knowledge service.
        
    Returns:
        Created knowledge entry.
    """
    try:
        entry = await knowledge_service.create_entry(
            knowledge_type=request.knowledge_type,
            statement=request.statement,
            domain=request.domain,
            metadata=request.metadata
        )
        
        return KnowledgeEntryResponse(**entry)
    except Exception as e:
        logger.error(f"Error creating knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge entry: {str(e)}")

@router.put("/entries/{entry_id}", response_model=KnowledgeEntryResponse)
async def update_knowledge_entry(
    entry_id: str,
    request: KnowledgeEntryRequest = Body(...),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Update a knowledge entry.
    
    Args:
        entry_id: Knowledge entry ID.
        request: Knowledge entry request.
        knowledge_service: Knowledge service.
        
    Returns:
        Updated knowledge entry.
    """
    try:
        entry = await knowledge_service.update_entry(
            entry_id=entry_id,
            knowledge_type=request.knowledge_type,
            statement=request.statement,
            domain=request.domain,
            metadata=request.metadata
        )
        
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Knowledge entry {entry_id} not found")
        
        return KnowledgeEntryResponse(**entry)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update knowledge entry: {str(e)}")

@router.delete("/entries/{entry_id}", response_model=Dict[str, Any])
async def delete_knowledge_entry(
    entry_id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Delete a knowledge entry.
    
    Args:
        entry_id: Knowledge entry ID.
        knowledge_service: Knowledge service.
        
    Returns:
        Deletion result.
    """
    try:
        success = await knowledge_service.delete_entry(entry_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Knowledge entry {entry_id} not found")
        
        return {
            "entry_id": entry_id,
            "deleted": True,
            "message": "Knowledge entry deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge entry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge entry: {str(e)}")