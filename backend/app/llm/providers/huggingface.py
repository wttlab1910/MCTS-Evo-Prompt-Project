"""
HuggingFace interface for LLM interactions.
"""
from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from app.llm.interface import LLMInterface
from app.utils.logger import get_logger
from app.utils.timer import timed

logger = get_logger("llm.huggingface")

class HuggingFaceInterface(LLMInterface):
    """
    Interface for HuggingFace models.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize HuggingFace interface.
        
        Args:
            model_name: Name of the HuggingFace model.
            **kwargs: Additional parameters for the model.
        """
        logger.info(f"Initializing HuggingFace interface for model: {model_name}")
        
        self.model_name = model_name
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = kwargs.get("max_length", 1024)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
                
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            logger.info(f"HuggingFace model {model_name} loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_name}: {e}")
            raise
    
    @timed
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt text.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated response text.
        """
        # Override default parameters with kwargs
        max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = response[0]["generated_text"]
            
            # Remove prompt from the beginning if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    @timed
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompt texts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated response texts.
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "device": self.device,
            "parameters": {
                "max_length": self.max_length,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        }