"""
Error feedback module.

This module provides functionality for collecting, analyzing, and generating
feedback from error patterns.
"""

from app.knowledge.error.error_collector import ErrorCollector
from app.knowledge.error.error_analyzer import ErrorAnalyzer
from app.knowledge.error.feedback_generator import FeedbackGenerator

__all__ = [
    'ErrorCollector',
    'ErrorAnalyzer',
    'FeedbackGenerator'
]