"""
Logging utility.
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.config import LOG_DIR

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Maximum log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024
# Number of backup files
BACKUP_COUNT = 5

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name.
        
    Returns:
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set log level
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)
        
        # Create file handler
        file_handler = RotatingFileHandler(
            LOG_DIR / f"{name.replace('.', '_')}.log",
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    
    return logger