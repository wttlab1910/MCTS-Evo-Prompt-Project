"""
Logging utility for MCTS-Evo-Prompt system.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os

# 设置默认日志级别和位置
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "False").lower() == "true"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "mcts_evo_prompt.log"

def setup_logging():
    """
    Set up logging configuration.
    Creates a logger with both console and file handlers.
    
    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("mcts_evo_prompt")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if LOG_TO_FILE:
        # Ensure log directory exists
        LOG_DIR.mkdir(exist_ok=True, parents=True)
        
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10485760, backupCount=5  # 10 MB
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """
    Get a named logger.
    
    Args:
        name: Name of the logger, typically the module name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(f"mcts_evo_prompt.{name}")