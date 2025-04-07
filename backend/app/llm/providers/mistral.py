"""
Mistral model provider.
"""
from typing import Dict, Any, List, Optional, Union
import os
import time
import asyncio
from app.llm.interface import LLMInterface
from app.utils.logger import get_logger

logger = get_logger("llm.providers.mistral")

class MistralProvider(LLMInterface):
    """
    Provider for Mistral models.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the Mistral provider.
        
        Args:
            model_id: Identifier for the model to use.
            **kwargs: Additional parameters:
                - local_path: Path to local model (optional).
                - max_tokens: Maximum number of tokens to generate.
                - temperature: Sampling temperature.
                - timeout: Timeout in seconds.
        """
        super().__init__(model_id, **kwargs)
        self.local_path = kwargs.get("local_path")
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.temperature = kwargs.get("temperature", 0.7)
        self.timeout = kwargs.get("timeout", 30)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Mistral model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            logger.info(f"Loading Mistral model: {self.model_id}")
            
            # Check for GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load tokenizer and model
            model_path = self.local_path if self.local_path else self.model_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
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
            
            logger.info(f"Successfully loaded Mistral model: {self.model_id}")
        except ImportError as e:
            logger.error(f"Failed to import required packages for Mistral model: {e}")
            raise ImportError("Please install transformers, torch, and accelerate to use Mistral models.")
        except Exception as e:
            logger.error(f"Failed to load Mistral model: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Dictionary containing generated text and metadata.
        """
        # Get parameters, override defaults if provided
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        timeout = kwargs.get("timeout", self.timeout)
        
        start_time = time.time()
        
        try:
            # Format prompt for Mistral
            # Mistral models use a specific format for instructions
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._generate_sync, formatted_prompt, max_tokens, temperature
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
            prompt: The formatted input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text.
        """
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
        
        # Strip the prompt and instruction tags
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]", 1)[1].strip()
        elif generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
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