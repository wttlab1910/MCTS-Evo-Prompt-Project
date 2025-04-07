"""
Authentication middleware.
"""
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import os
from app.utils.logger import get_logger

logger = get_logger("api.middleware.auth")

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API authentication.
    
    This middleware implements a simple API key authentication mechanism.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize the authentication middleware.
        
        Args:
            app: ASGI application.
        """
        super().__init__(app)
        self.api_key = os.environ.get("API_KEY", "")
        self.skip_auth = not self.api_key
        
        if self.skip_auth:
            logger.warning("API key not set. Authentication is disabled.")
        else:
            logger.info("Authentication middleware initialized.")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request.
        
        Args:
            request: HTTP request.
            call_next: Next middleware or route handler.
            
        Returns:
            HTTP response.
        """
        # Skip authentication if API key is not set
        if self.skip_auth:
            return await call_next(request)
        
        # Skip authentication for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        # Get API key from header or query parameter
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            api_key = request.query_params.get("api_key")
        
        # Validate API key
        if not api_key or api_key != self.api_key:
            logger.warning(f"Unauthorized access attempt from {request.client.host}")
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        
        # Proceed with the request
        return await call_next(request)