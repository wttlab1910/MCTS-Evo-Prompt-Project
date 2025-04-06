"""
Main entry point for the MCTS-Evo-Prompt application.
"""
from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from app.api.routes import router as api_router
from app.api.middleware.logging import LoggingMiddleware
from app.config import API_PREFIX
from app.utils.logger import setup_logging

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI(
    title="MCTS-Evo-Prompt",
    description="Strategic Planning Framework for Expert-Level Prompt Optimization",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Register API routes
app.include_router(api_router, prefix=API_PREFIX)

@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {
        "message": "Welcome to MCTS-Evo-Prompt API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }

if __name__ == "__main__":
    import uvicorn # type: ignore
    from app.config import API_HOST, API_PORT

    logger.info(f"Starting MCTS-Evo-Prompt API on {API_HOST}:{API_PORT}")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)