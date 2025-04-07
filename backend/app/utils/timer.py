"""
Timer utility for measuring execution time.
"""
import time
import functools
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar
from app.utils.logger import get_logger

logger = get_logger("utils.timer")

T = TypeVar('T')

class Timer:
    """
    Timer context manager and decorator for measuring execution time.
    """
    
    def __init__(self, name: str, log_level: str = "debug"):
        """
        Initialize the timer.
        
        Args:
            name: Timer name for logging.
            log_level: Log level for reporting elapsed time.
        """
        self.name = name
        self.log_level = log_level.lower()
        self.start_time = 0
        self.elapsed = 0
        
    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log elapsed time."""
        self.elapsed = time.time() - self.start_time
        
        # Log elapsed time
        log_message = f"Timer '{self.name}' elapsed time: {self.elapsed:.6f}s"
        
        if self.log_level == "debug":
            logger.debug(log_message)
        elif self.log_level == "info":
            logger.info(log_message)
        elif self.log_level == "warning":
            logger.warning(log_message)
        
    @property
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds.
        """
        return self.elapsed * 1000


def timed(name: Optional[str] = None, log_level: str = "debug"):
    """
    Decorator for measuring function execution time.
    
    Args:
        name: Timer name for logging.
        log_level: Log level for reporting elapsed time.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            
            with Timer(timer_name, log_level=log_level):
                return func(*args, **kwargs)
            
        return wrapper
    
    return decorator


class TimingStats:
    """
    Collect and report timing statistics.
    """
    
    def __init__(self):
        """Initialize the timing stats collector."""
        self.stats: Dict[str, List[float]] = {}
    
    def record(self, name: str, elapsed: float) -> None:
        """
        Record a timing measurement.
        
        Args:
            name: Operation name.
            elapsed: Elapsed time in seconds.
        """
        if name not in self.stats:
            self.stats[name] = []
        
        self.stats[name].append(elapsed)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation.
        
        Args:
            name: Operation name.
            
        Returns:
            Dictionary with statistics.
        """
        if name not in self.stats or not self.stats[name]:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "total": 0
            }
        
        measurements = self.stats[name]
        
        return {
            "count": len(measurements),
            "min": min(measurements),
            "max": max(measurements),
            "avg": sum(measurements) / len(measurements),
            "total": sum(measurements)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operations.
        
        Returns:
            Dictionary with statistics for all operations.
        """
        return {name: self.get_stats(name) for name in self.stats}
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset timing statistics.
        
        Args:
            name: Operation name to reset. If None, reset all statistics.
        """
        if name is not None:
            if name in self.stats:
                self.stats[name] = []
        else:
            self.stats = {}


# Global timing stats instance
timing_stats = TimingStats()