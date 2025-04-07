"""
HuggingFace model provider.
"""
from typing import Dict, Any, List, Optional, Union
import os
import time
import asyncio
from app.llm.interface import LLMInterface
from app.utils.logger import get_logger

logger = get_logger("llm.providers.huggingface")

class HuggingFaceProvider(LLMInterface):
    """
    Provider for HuggingFace models.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the HuggingFace provider.
        
        Args:
            model_id: Identifier for the model to use.
            **kwargs: Additional parameters:
                - api_key: HuggingFace API key (optional for some models).
                - max_tokens: Maximum number of tokens to generate.
                - temperature: Sampling temperature.
                - timeout: Timeout in seconds.
        """
        super().__init__(model_id, **kwargs)
        self.api_key = kwargs.get("api_key", os.environ.get("HUGGINGFACE_API_KEY", ""))
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.temperature = kwargs.get("temperature", 0.7)
        self.timeout = kwargs.get("timeout", 30)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the model either via API or locally."""
        try:
            if self.api_key:
                # Use HuggingFace Inference API
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=self.api_key)
                logger.info(f"Using HuggingFace Inference API for model: {self.model_id}")
            else:
                # Load model locally
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                    
                    logger.info(f"Loading local model: {self.model_id}")
                    
                    # Check for GPU availability
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Using device: {device}")
                    
                    # Load tokenizer and model
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map="auto" if device == "cuda" else None
                    )
                    
                    # Create generation pipeline
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device
                    )
                    
                    logger.info(f"Successfully loaded local model: {self.model_id}")
                except ImportError as e:
                    logger.error(f"Failed to import required packages for local model: {e}")
                    raise ImportError("Please install transformers, torch, and accelerate to use local models.")
                except Exception as e:
                    logger.error(f"Failed to load local model: {e}")
                    raise
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace provider: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters:
                - max_tokens: Maximum number of tokens to generate.
                - temperature: Sampling temperature.
                - timeout: Timeout in seconds.
            
        Returns:
            Dictionary containing generated text and metadata.
        """
        # Get parameters, override defaults if provided
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        timeout = kwargs.get("timeout", self.timeout)
        
        start_time = time.time()
        
        try:
            # Run in executor to avoid blocking the event loop
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt, max_tokens, temperature
            )
            
            elapsed_time = time.time() - start_time
            
            return {
                "text": result,
                "prompt": prompt,
                "model": self.model_id,
                "elapsed_time": elapsed_time,
                "finish_reason": "stop"
            }
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            logger.warning(f"Generation timed out after {elapsed_time:.2f}s")
            
            return {
                "text": "",
                "prompt": prompt,
                "model": self.model_id,
                "elapsed_time": elapsed_time,
                "finish_reason": "timeout",
                "error": "Request timed out"
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Generation failed: {e}")
            
            return {
                "text": "",
                "prompt": prompt,
                "model": self.model_id,
                "elapsed_time": elapsed_time,
                "finish_reason": "error",
                "error": str(e)
            }
    
    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Synchronous text generation.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text.
        """
        if hasattr(self, "client"):
            # Use Inference API
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )
            return response
        else:
            # Use local model
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "top_p": 0.9,
                "top_k": 50,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            outputs = self.pipeline(
                prompt,
                **generation_config
            )
            
            # Extract generated text, removing the prompt
            generated_text = outputs[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
                
            return generated_text
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of dictionaries containing generated text and metadata.
        """
        # Create tasks for each prompt
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        
        # Run all tasks concurrently
        return await asyncio.gather(*tasks)