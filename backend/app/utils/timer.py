"""
Timer utility for measuring execution time.
"""
import time
from functools import wraps
from app.utils.logger import get_logger

logger = get_logger("timer")

class Timer:
    """
    Context manager for measuring execution time.
    """
    
    def __init__(self, name="Operation"):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed.
        """
        self.name = name
        
    def __enter__(self):
        """
        Start timer when entering context.
        """
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log execution time when exiting context.
        """
        elapsed_time = time.time() - self.start_time
        logger.info(f"{self.name} completed in {elapsed_time:.4f} seconds")

def timed(func):
    """
    Decorator for measuring function execution time.
    
    Args:
        func: Function to be timed.
        
    Returns:
        Wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper