"""
Knowledge integration module.

This module provides functionality for integrating domain knowledge into prompts,
including selection of integration strategies and formatting of knowledge elements.
"""

from app.knowledge.integration.integrator import (
    KnowledgeIntegrator,
    PromptKnowledgeIntegrator
)
from app.knowledge.integration.strategy import (
    IntegrationStrategy,
    PlacementStrategy,
    FormatSelectionStrategy,
    ConflictResolutionStrategy,
    TemplateIntegrationStrategy
)

__all__ = [
    'KnowledgeIntegrator',
    'PromptKnowledgeIntegrator',
    'IntegrationStrategy',
    'PlacementStrategy',
    'FormatSelectionStrategy',
    'ConflictResolutionStrategy',
    'TemplateIntegrationStrategy'
]