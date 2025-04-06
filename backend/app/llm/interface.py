"""
Base interface for LLM interactions.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from app.utils.logger import get_logger

logger = get_logger("llm.interface")

class LLMInterface(ABC):
    """
    Abstract base class for LLM interfaces.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt text.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated response text.
        """
        pass
        
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompt texts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated response texts.
        """
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information.
        """
        pass

class LLMFactory:
    """
    Factory class for creating LLM interfaces.
    """
    
    @staticmethod
    def create(provider: str, model_name: str, **kwargs) -> LLMInterface:
        """
        Create an LLM interface for the specified provider and model.
        
        Args:
            provider: LLM provider (e.g., "huggingface", "mistral", "gemma").
            model_name: Name of the model.
            **kwargs: Additional parameters for the interface.
            
        Returns:
            LLM interface instance.
            
        Raises:
            ValueError: If the provider is not supported.
        """
        if provider == "huggingface":
            from app.llm.providers.huggingface import HuggingFaceInterface
            return HuggingFaceInterface(model_name, **kwargs)
        elif provider == "mistral":
            from app.llm.providers.mistral import MistralInterface
            return MistralInterface(model_name, **kwargs)
        elif provider == "gemma":
            from app.llm.providers.gemma import GemmaInterface
            return GemmaInterface(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")