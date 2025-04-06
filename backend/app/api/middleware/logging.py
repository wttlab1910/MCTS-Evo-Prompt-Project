"""
Logging middleware for API requests and responses.
"""
import time
import json
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from app.utils.logger import get_logger

logger = get_logger("middleware.logging")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process an incoming request and log details.
        
        Args:
            request: The incoming request.
            call_next: Function to call the next middleware or route handler.
            
        Returns:
            Response from the next middleware or route handler.
        """
        # Log request
        start_time = time.time()
        
        # Get request body
        request_body = b""
        if request.method in ["POST", "PUT", "PATCH"]:
            request_body = await request.body()
            # Store the body so it can be read again by the route handler
            await request._body_reset(request_body)
            
        # Try to parse and log request body
        try:
            if request_body:
                body = json.loads(request_body)
                # Truncate large request bodies
                if isinstance(body, dict) and len(str(body)) > 1000:
                    body = {k: (str(v)[:100] + "..." if len(str(v)) > 100 else v) for k, v in body.items()}
            else:
                body = None
        except Exception:
            body = "<non-JSON body>"
            
        # Log request details
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"Params: {dict(request.query_params)} "
            f"Body: {body}"
        )
        
        # Process request and get response
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            logger.error(f"Request failed: {str(e)}")
            raise
            
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Time: {process_time:.4f}s"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response