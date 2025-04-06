"""
Authentication middleware for API routes.
"""
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from app.config import AUTH_REQUIRED, AUTH_TOKEN
from app.utils.logger import get_logger

logger = get_logger("middleware.auth")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(request: Request, api_key: str = Depends(api_key_header)):
    """
    Verify API key for protected routes.
    
    Args:
        request: The incoming request.
        api_key: API key from the header.
        
    Raises:
        HTTPException: If authentication fails.
    """
    if not AUTH_REQUIRED:
        return True
        
    if api_key != AUTH_TOKEN:
        logger.warning(f"Invalid API key attempt: {api_key}")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    return True