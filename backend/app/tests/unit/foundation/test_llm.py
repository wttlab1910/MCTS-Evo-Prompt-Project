"""
Unit tests for LLM interface components.
"""
import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock
from app.llm.interface import LLMInterface, LLMFactory
from app.llm.providers.huggingface import HuggingFaceProvider
from app.llm.providers.mistral import MistralProvider
from app.llm.providers.gemma import GemmaProvider
from app.llm.optimization.caching import LLMResponseCache
from app.llm.optimization.batching import BatchProcessor
from app.config import RESPONSES_CACHE_DIR

class MockProvider(LLMInterface):
    """Mock provider for testing."""
    
    async def generate(self, prompt, **kwargs):
        return {
            "text": f"Response to: {prompt}",
            "prompt": prompt,
            "model": self.model_id,
            "elapsed_time": 0.1,
            "finish_reason": "stop"
        }
    
    async def generate_batch(self, prompts, **kwargs):
        return [await self.generate(prompt, **kwargs) for prompt in prompts]

class TestLLMInterface:
    """Tests for the LLM interface."""
    
    def setup_method(self):
        """Set up test environment."""
        # Register mock provider
        LLMFactory.register_provider("mock", MockProvider)
    
    def test_factory_registration(self):
        """Test provider registration with factory."""
        assert "mock" in LLMFactory._providers
        assert "huggingface" in LLMFactory._providers
        assert "mistral" in LLMFactory._providers
        assert "gemma" in LLMFactory._providers
    
    def test_factory_creation(self):
        """Test factory creates appropriate provider."""
        llm = LLMFactory.create("mock", "test-model")
        assert isinstance(llm, MockProvider)
        assert llm.model_id == "test-model"
    
    def test_factory_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        with pytest.raises(ValueError):
            LLMFactory.create("unknown", "test-model")
    
    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generate method."""
        llm = LLMFactory.create("mock", "test-model")
        response = await llm.generate("Hello")
        
        assert response["text"] == "Response to: Hello"
        assert response["prompt"] == "Hello"
        assert response["model"] == "test-model"
        assert response["finish_reason"] == "stop"
    
    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """Test generate_batch method."""
        llm = LLMFactory.create("mock", "test-model")
        prompts = ["Hello", "World"]
        responses = await llm.generate_batch(prompts)
        
        assert len(responses) == 2
        assert responses[0]["text"] == "Response to: Hello"
        assert responses[1]["text"] == "Response to: World"
    
    def test_synchronous_methods(self):
        """Test synchronous wrapper methods."""
        llm = LLMFactory.create("mock", "test-model")
        
        # Use event loop directly to avoid pytest issues
        response = asyncio.get_event_loop().run_until_complete(llm.generate("Hello"))
        assert response["text"] == "Response to: Hello"
        
        responses = asyncio.get_event_loop().run_until_complete(llm.generate_batch(["Hello", "World"]))
        assert len(responses) == 2


class TestLLMCaching:
    """Tests for LLM response caching."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cache = LLMResponseCache()
        
        # Clear cache directory
        for file in RESPONSES_CACHE_DIR.glob("*.json"):
            os.remove(file)
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = self.cache._get_cache_key("Hello", "model1")
        key2 = self.cache._get_cache_key("Hello", "model2")
        key3 = self.cache._get_cache_key("Hello", "model1", max_tokens=100)
        
        assert key1 != key2  # Different models should have different keys
        assert key1 != key3  # Different parameters should have different keys
    
    def test_cache_set_get(self):
        """Test setting and getting cached responses."""
        response = {"text": "Hello response", "elapsed_time": 0.1}
        
        # Set in cache
        self.cache.set("Hello", "test-model", response)
        
        # Get from cache
        cached = self.cache.get("Hello", "test-model")
        assert cached is not None
        assert cached["text"] == "Hello response"
    
    def test_cache_invalidation(self):
        """Test invalidating cached responses."""
        response = {"text": "Hello response", "elapsed_time": 0.1}
        
        # Set in cache
        self.cache.set("Hello", "test-model", response)
        
        # Invalidate
        success = self.cache.invalidate("Hello", "test-model")
        assert success
        
        # Try to get after invalidation
        cached = self.cache.get("Hello", "test-model")
        assert cached is None


class TestBatchProcessor:
    """Tests for batch processing."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing."""
        # Mock processor function
        async def process_batch(items):
            return [f"Processed: {item}" for item in items]
        
        # Create batch processor
        processor = BatchProcessor(
            batch_size=3,
            timeout=1.0,
            processor_fn=process_batch
        )
        
        # Process items
        results = await asyncio.gather(
            processor.process("item1"),
            processor.process("item2"),
            processor.process("item3")
        )
        
        assert results == ["Processed: item1", "Processed: item2", "Processed: item3"]
    
    @pytest.mark.asyncio
    async def test_batch_timeout(self):
        """Test batch processing with timeout."""
        # Mock processor function
        async def process_batch(items):
            return [f"Processed: {item}" for item in items]
        
        # Create batch processor with short timeout
        processor = BatchProcessor(
            batch_size=3,
            timeout=0.1,
            processor_fn=process_batch
        )
        
        # Process items with delay between them
        task1 = asyncio.create_task(processor.process("item1"))
        await asyncio.sleep(0.2)  # Wait for timeout to trigger
        task2 = asyncio.create_task(processor.process("item2"))
        
        results = await asyncio.gather(task1, task2)
        
        assert results[0] == "Processed: item1"
        assert results[1] == "Processed: item2"