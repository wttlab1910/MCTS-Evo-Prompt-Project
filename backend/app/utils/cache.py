"""
Cache management utility.
"""
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
import time
import functools
from app.utils.logger import get_logger

logger = get_logger("utils.cache")

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class MemoryCache(Generic[K, V]):
    """
    Simple in-memory cache with expiration.
    """
    
    def __init__(self, expiration: int = 3600):
        """
        Initialize the cache.
        
        Args:
            expiration: Cache expiration time in seconds (default: 1 hour).
        """
        self.cache: Dict[K, Dict[str, Any]] = {}
        self.expiration = expiration
        
        logger.debug(f"Initialized MemoryCache with expiration={expiration}s")
    
    def get(self, key: K) -> Optional[V]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None if not found or expired.
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry["expires"] < time.time():
            self.cache.pop(key)
            return None
        
        return entry["value"]
    
    def set(self, key: K, value: V, expiration: Optional[int] = None) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            expiration: Custom expiration time in seconds.
        """
        expires = time.time() + (expiration if expiration is not None else self.expiration)
        
        self.cache[key] = {
            "value": value,
            "expires": expires
        }
    
    def delete(self, key: K) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key.
            
        Returns:
            True if key was deleted, False if not found.
        """
        if key in self.cache:
            self.cache.pop(key)
            return True
        
        return False
    
    def clear(self) -> None:
        """
        Clear all entries from the cache.
        """
        self.cache.clear()
    
    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed.
        """
        now = time.time()
        expired_keys = [k for k, v in self.cache.items() if v["expires"] < now]
        
        for key in expired_keys:
            self.cache.pop(key)
        
        return len(expired_keys)


def memoize(expiration: int = 3600):
    """
    Decorator for memoizing function results.
    
    Args:
        expiration: Cache expiration time in seconds (default: 1 hour).
        
    Returns:
        Decorated function.
    """
    cache = MemoryCache(expiration=expiration)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = hash(tuple(key_parts))
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, result)
            
            return result
            
        return wrapper
    
    return decorator

# Add cached as an alias for memoize for backward compatibility
cached = memoize