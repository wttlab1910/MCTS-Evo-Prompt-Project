"""
API routes for MCTS-Evo-Prompt system.
"""
from fastapi import APIRouter
from app.api.endpoints import prompt_routes, optimization_routes, knowledge_routes
from app.utils.logger import get_logger

logger = get_logger("api.routes")

# Create router
router = APIRouter()

# Include all endpoint routers
router.include_router(prompt_routes.router, prefix="/prompts", tags=["Prompts"])
router.include_router(optimization_routes.router, prefix="/optimization", tags=["Optimization"])
router.include_router(knowledge_routes.router, prefix="/knowledge", tags=["Knowledge"])

logger.info("API routes registered")