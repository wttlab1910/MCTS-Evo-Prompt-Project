"""
Unit tests for utility functions.
"""
import pytest
import os
import time
import json
from datetime import datetime
from pathlib import Path
from app.utils.logger import get_logger
from app.utils.cache import MemoryCache, memoize
from app.utils.timer import Timer, timed, timing_stats
from app.utils.serialization import to_json, from_json, save_json, load_json, to_dataclass
from dataclasses import dataclass

# 为每个测试函数设置 setup/teardown，确保 timing_stats 被重置
@pytest.fixture(autouse=True)
def reset_timing_stats():
    timing_stats.reset()
    yield
    timing_stats.reset()

class TestLogger:
    """Tests for the logger utility."""
    
    def test_get_logger(self):
        """Test creating a logger."""
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        
        # Test that the logger is properly configured
        assert len(logger.handlers) > 0
        
        # Test logging (no exceptions should be raised)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

class TestMemoryCache:
    """Tests for the memory cache."""
    
    def test_cache_set_get(self):
        """Test setting and getting cached values."""
        cache = MemoryCache()
        
        # Set value
        cache.set("key1", "value1")
        
        # Get value
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = MemoryCache(expiration=0.1)
        
        # Set value
        cache.set("key1", "value1")
        
        # Value should exist initially
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Value should be expired
        assert cache.get("key1") is None
    
    def test_cache_delete(self):
        """Test deleting from cache."""
        cache = MemoryCache()
        
        # Set value
        cache.set("key1", "value1")
        
        # Delete
        success = cache.delete("key1")
        assert success
        
        # Value should be gone
        assert cache.get("key1") is None
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = MemoryCache()
        
        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Clear
        cache.clear()
        
        # Values should be gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_memoize_decorator(self):
        """Test memoize decorator."""
        call_count = 0
        
        @memoize(expiration=1)
        def test_function(arg):
            nonlocal call_count
            call_count += 1
            return f"Result {arg}"
        
        # First call should increment counter
        result1 = test_function("test")
        assert result1 == "Result test"
        assert call_count == 1
        
        # Second call should use cache
        result2 = test_function("test")
        assert result2 == "Result test"
        assert call_count == 1
        
        # Different arg should increment counter
        result3 = test_function("other")
        assert result3 == "Result other"
        assert call_count == 2

class TestTimer:
    """Tests for the timer utility."""
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with Timer("test_timer") as timer:
            time.sleep(0.1)
        
        assert timer.elapsed > 0
        assert timer.elapsed_ms > 0
    
    def test_timed_decorator(self):
        """Test timed decorator."""
        @timed("test_function")
        def test_function():
            time.sleep(0.1)
            return "result"
        
        result = test_function()
        assert result == "result"
    
    def test_timing_stats(self):
        """Test timing stats collection."""
        # Record some timings
        timing_stats.record("operation1", 0.1)
        timing_stats.record("operation1", 0.2)
        timing_stats.record("operation2", 0.3)
        
        # Get stats for operation1
        stats1 = timing_stats.get_stats("operation1")
        assert stats1["count"] == 2
        assert stats1["min"] == 0.1
        assert stats1["max"] == 0.2
        assert round(stats1["avg"], 2) == 0.15  # Round to avoid floating point issues
        assert round(stats1["total"], 2) == 0.3  # 同样对total使用round
        
        # Get all stats
        all_stats = timing_stats.get_all_stats()
        assert "operation1" in all_stats
        assert "operation2" in all_stats
        
        # Reset specific operation
        timing_stats.reset("operation1")
        assert timing_stats.get_stats("operation1")["count"] == 0
        assert timing_stats.get_stats("operation2")["count"] == 1
        
        # Reset all
        timing_stats.reset()
        assert timing_stats.get_stats("operation2")["count"] == 0

class TestSerialization:
    """Tests for the serialization utility."""
    
    def test_to_from_json(self):
        """Test converting to and from JSON."""
        data = {"name": "Test", "value": 123, "nested": {"key": "value"}}
        
        # Convert to JSON
        json_str = to_json(data)
        
        # Convert back from JSON
        result = from_json(json_str)
        
        assert result == data
    
    def test_save_load_json(self):
        """Test saving and loading JSON to/from file."""
        data = {"name": "Test", "value": 123}
        file_path = Path("test_data.json")
        
        # Save to file
        save_json(data, file_path)
        
        # Load from file
        result = load_json(file_path)
        
        assert result == data
        
        # Clean up
        if file_path.exists():
            os.remove(file_path)
    
    def test_datetime_serialization(self):
        """Test serialization of datetime objects."""
        data = {"time": datetime(2023, 1, 1, 12, 0, 0)}
        
        # Convert to JSON
        json_str = to_json(data)
        
        # Datetime should be converted to ISO format
        assert "2023-01-01T12:00:00" in json_str
    
    def test_to_dataclass(self):
        """Test converting dictionary to dataclass."""
        @dataclass
        class TestData:
            name: str
            value: int
        
        data = {"name": "Test", "value": 123, "extra": "ignored"}
        
        # Convert to dataclass
        result = to_dataclass(data, TestData)
        
        assert isinstance(result, TestData)
        assert result.name == "Test"
        assert result.value == 123

# 修复独立的测试函数
def test_timing_stats():
    """Test timing stats collection."""
    # Reset stats to ensure clean state
    timing_stats.reset()
    
    # Record some timings
    timing_stats.record("operation1", 0.1)
    timing_stats.record("operation1", 0.2)
    timing_stats.record("operation2", 0.3)
    
    # Get stats for operation1
    stats1 = timing_stats.get_stats("operation1")
    assert stats1["count"] == 2
    assert stats1["min"] == 0.1
    assert stats1["max"] == 0.2
    assert round(stats1["avg"], 2) == 0.15  # Round to avoid floating point issues
    assert round(stats1["total"], 2) == 0.3  # 同样对total使用round
    
    # Get all stats
    all_stats = timing_stats.get_all_stats()
    assert "operation1" in all_stats
    assert "operation2" in all_stats
    
    # Reset specific operation
    timing_stats.reset("operation1")
    assert timing_stats.get_stats("operation1")["count"] == 0
    assert timing_stats.get_stats("operation2")["count"] == 1
    
    # Reset all
    timing_stats.reset()
    assert timing_stats.get_stats("operation2")["count"] == 0