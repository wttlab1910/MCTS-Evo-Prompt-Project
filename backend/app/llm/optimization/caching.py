"""
Caching for LLM responses.
"""
from typing import Dict, Any, List, Optional, Union, Type, TypeVar, Generic
import os
import json
import hashlib
import time
from pathlib import Path
from app.utils.logger import get_logger
from app.config import RESPONSES_CACHE_DIR

logger = get_logger("llm.optimization.caching")

T = TypeVar('T')

class LLMResponseCache(Generic[T]):
    """
    Cache for LLM responses.
    
    This class provides a persistent cache for LLM responses to avoid redundant API calls.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, expiration: int = 86400):
        """
        Initialize the response cache.
        
        Args:
            cache_dir: Directory to store cache files.
            expiration: Cache expiration time in seconds (default: 24 hours).
        """
        self.cache_dir = cache_dir or RESPONSES_CACHE_DIR
        self.expiration = expiration
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.debug(f"Initialized LLMResponseCache at {self.cache_dir}")
    
    def _get_cache_key(self, prompt: str, model_id: str, **kwargs) -> str:
        """
        Generate a cache key for a prompt and parameters.
        
        Args:
            prompt: The input prompt.
            model_id: The model identifier.
            **kwargs: Additional generation parameters.
            
        Returns:
            Cache key string.
        """
        # Create a string containing all parameters
        param_str = prompt + model_id
        
        # Add significant parameters to the key
        significant_params = ["max_tokens", "temperature", "top_p", "top_k"]
        for param in significant_params:
            if param in kwargs:
                param_str += f"_{param}={kwargs[param]}"
        
        # Hash the parameter string
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: The cache key.
            
        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, prompt: str, model_id: str, **kwargs) -> Optional[T]:
        """
        Get a cached response if available.
        
        Args:
            prompt: The input prompt.
            model_id: The model identifier.
            **kwargs: Additional generation parameters.
            
        Returns:
            Cached response or None if not found.
        """
        cache_key = self._get_cache_key(prompt, model_id, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Read cache file
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # Check expiration
            if cache_data.get("timestamp", 0) + self.expiration < time.time():
                logger.debug(f"Cache expired for key {cache_key}")
                return None
            
            logger.debug(f"Cache hit for key {cache_key}")
            return cache_data.get("response")
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            return None
    
    def set(self, prompt: str, model_id: str, response: T, **kwargs) -> None:
        """
        Store a response in the cache.
        
        Args:
            prompt: The input prompt.
            model_id: The model identifier.
            response: The response to cache.
            **kwargs: Additional generation parameters.
        """
        cache_key = self._get_cache_key(prompt, model_id, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Create cache data
            cache_data = {
                "timestamp": time.time(),
                "prompt": prompt,
                "model_id": model_id,
                "params": {k: v for k, v in kwargs.items() if k not in ["api_key"]},
                "response": response
            }
            
            # Write cache file
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Cached response for key {cache_key}")
        except Exception as e:
            logger.warning(f"Error writing cache file {cache_path}: {e}")
    
    def invalidate(self, prompt: str, model_id: str, **kwargs) -> bool:
        """
        Invalidate a cached response.
        
        Args:
            prompt: The input prompt.
            model_id: The model identifier.
            **kwargs: Additional generation parameters.
            
        Returns:
            True if the cache was invalidated, False otherwise.
        """
        cache_key = self._get_cache_key(prompt, model_id, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return False
        
        try:
            # Remove cache file
            os.remove(cache_path)
            logger.debug(f"Invalidated cache for key {cache_key}")
            return True
        except Exception as e:
            logger.warning(f"Error invalidating cache file {cache_path}: {e}")
            return False
    
    def clear(self, older_than: Optional[int] = None) -> int:
        """
        Clear all or expired cache entries.
        
        Args:
            older_than: Clear entries older than this many seconds.
                If None, clear all entries.
                
        Returns:
            Number of entries cleared.
        """
        cleared = 0
        
        try:
            # Iterate over all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                if older_than is not None:
                    # Check file age
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            cache_data = json.load(f)
                        
                        # Skip if not expired
                        if cache_data.get("timestamp", 0) > time.time() - older_than:
                            continue
                    except Exception:
                        # If we can't read the file, assume it's invalid and remove it
                        pass
                
                # Remove cache file
                os.remove(cache_file)
                cleared += 1
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
        
        logger.debug(f"Cleared {cleared} cache entries")
        return cleared