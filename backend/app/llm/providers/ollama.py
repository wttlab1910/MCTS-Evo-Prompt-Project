"""
Ollama model provider.
"""
import aiohttp
import time
import asyncio
from typing import Dict, Any, List, Optional
from app.llm.interface import LLMInterface
from app.utils.logger import get_logger

logger = get_logger("llm.providers.ollama")

class OllamaProvider(LLMInterface):
    """
    Provider for Ollama models.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the Ollama provider.
        
        Args:
            model_id: Model name in Ollama (e.g., "mistral", "gemma3:12b", "deepseek-r1:32b")
            **kwargs: Additional parameters
        """
        super().__init__(model_id, **kwargs)
        self.api_base = kwargs.get("api_base", "http://localhost:11434")
        self.timeout = kwargs.get("timeout", 60)
        
        logger.info(f"Initialized Ollama provider for model: {model_id}")
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using Ollama API.
        """
        start_time = time.time()
        
        # Prepare request data
        request_data = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 2048)
            }
        }
        
        api_url = f"{self.api_base}/api/generate"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url, 
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama API error: {response.status}, {error_text}")
                    
                    result = await response.json()
                    
                    elapsed_time = time.time() - start_time
                    return {
                        "text": result.get("response", ""),
                        "prompt": prompt,
                        "model": self.model_id,
                        "elapsed_time": elapsed_time,
                        "finish_reason": "stop",
                        "raw_response": result
                    }
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error generating text with Ollama: {e}")
            return {
                "text": "",
                "prompt": prompt,
                "model": self.model_id,
                "elapsed_time": elapsed_time,
                "finish_reason": "error",
                "error": str(e)
            }
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.
        """
        # Create tasks for each prompt
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        
        # Run all tasks concurrently
        return await asyncio.gather(*tasks)