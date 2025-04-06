"""
Cache management utility for MCTS-Evo-Prompt system.
"""
import json
import time
import os
from pathlib import Path
from functools import wraps
from app.config import CACHE_DIR, CACHE_ENABLED, CACHE_EXPIRATION
from app.utils.logger import get_logger
from app.utils.serialization import dumps, loads

logger = get_logger("cache")

class Cache:
    """
    Simple file-based cache implementation.
    """
    
    def __init__(self, directory=CACHE_DIR, expiration=CACHE_EXPIRATION):
        """
        Initialize cache.
        
        Args:
            directory: Directory to store cache files.
            expiration: Cache expiration time in seconds.
        """
        self.directory = Path(directory)
        self.expiration = expiration
        self.directory.mkdir(exist_ok=True, parents=True)
        
    def get(self, key, default=None):
        """
        Get value from cache.
        
        Args:
            key: Cache key.
            default: Default value if key not found.
            
        Returns:
            Cached value or default.
        """
        if not CACHE_ENABLED:
            return default
            
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return default
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            # Check if cache has expired
            if time.time() - data['timestamp'] > self.expiration:
                logger.debug(f"Cache expired for key: {key}")
                self.delete(key)
                return default
                
            logger.debug(f"Cache hit for key: {key}")
            return loads(data['value'])
        except Exception as e:
            logger.warning(f"Error reading cache for key {key}: {e}")
            return default
            
    def set(self, key, value):
        """
        Set value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
        """
        if not CACHE_ENABLED:
            return
            
        cache_file = self._get_cache_file(key)
        
        try:
            data = {
                'timestamp': time.time(),
                'value': dumps(value)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
            logger.debug(f"Cache set for key: {key}")
        except Exception as e:
            logger.warning(f"Error setting cache for key {key}: {e}")
            
    def delete(self, key):
        """
        Delete value from cache.
        
        Args:
            key: Cache key.
        """
        if not CACHE_ENABLED:
            return
            
        cache_file = self._get_cache_file(key)
        
        if cache_file.exists():
            try:
                os.remove(cache_file)
                logger.debug(f"Cache deleted for key: {key}")
            except Exception as e:
                logger.warning(f"Error deleting cache for key {key}: {e}")
                
    def clear(self):
        """
        Clear all cache entries.
        """
        if not CACHE_ENABLED:
            return
            
        for file in self.directory.glob("*.cache"):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Error clearing cache file {file}: {e}")
                
        logger.info("Cache cleared")
        
    def _get_cache_file(self, key):
        """
        Get cache file path for key.
        
        Args:
            key: Cache key.
            
        Returns:
            Path to cache file.
        """
        # Create a filename-safe representation of the key
        safe_key = "".join(c if c.isalnum() else "_" for c in str(key))
        return self.directory / f"{safe_key}.cache"

# Global cache instance
cache = Cache()

def cached(expiration=None):
    """
    Decorator for caching function results.
    
    Args:
        expiration: Cache expiration time in seconds.
        
    Returns:
        Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not CACHE_ENABLED:
                return func(*args, **kwargs)
                
            # Create a unique key based on function name and arguments
            key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            result = cache.get(key)
            
            if result is None:
                # Cache miss, execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                cache.set(key, result)
                
            return result
        return wrapper
    return decorator