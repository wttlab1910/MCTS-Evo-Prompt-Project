"""
Logging middleware for API requests and responses.
"""
from typing import Callable
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.utils.logger import get_logger

logger = get_logger("api.middleware.logging")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging.
    
    This middleware logs information about incoming requests and outgoing responses.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize the logging middleware.
        
        Args:
            app: ASGI application.
        """
        super().__init__(app)
        logger.info("Logging middleware initialized.")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request.
        
        Args:
            request: HTTP request.
            call_next: Next middleware or route handler.
            
        Returns:
            HTTP response.
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log request
        start_time = time.time()
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"(took {process_time:.3f}s)"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Error {request_id}: {str(e)} "
                f"(took {process_time:.3f}s)"
            )
            raise