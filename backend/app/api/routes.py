"""
API route definitions.
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.auth import AuthMiddleware
from app.api.endpoints.prompt_routes import router as prompt_router
from app.api.endpoints.optimization_routes import router as optimization_router
from app.api.endpoints.knowledge_routes import router as knowledge_router

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="MCTS-Evo-Prompt API",
        description="API for prompt optimization using MCTS with evolutionary algorithms",
        version="0.1.0"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # Include routers
    app.include_router(prompt_router, prefix="/api/prompts", tags=["prompts"])
    app.include_router(optimization_router, prefix="/api/optimization", tags=["optimization"])
    app.include_router(knowledge_router, prefix="/api/knowledge", tags=["knowledge"])
    
    # Add health check route
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok"}
    
    return app