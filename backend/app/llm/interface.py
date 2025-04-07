"""
Base interface for LLM interactions.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import abc
from app.utils.logger import get_logger

logger = get_logger("llm.interface")

class LLMInterface(abc.ABC):
    """
    Abstract base class for LLM interactions.
    
    This class defines the interface that all LLM providers must implement.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the LLM interface.
        
        Args:
            model_id: Identifier for the model to use.
            **kwargs: Additional provider-specific parameters.
        """
        self.model_id = model_id
        self.params = kwargs
        logger.info(f"Initialized LLM interface for model: {model_id}")
    
    @abc.abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Dictionary containing generated text and metadata.
        """
        pass
    
    @abc.abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of dictionaries containing generated text and metadata.
        """
        pass
    
    def generate_sync(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Synchronous version of generate.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Dictionary containing generated text and metadata.
        """
        return asyncio.run(self.generate(prompt, **kwargs))
    
    def generate_batch_sync(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Synchronous version of generate_batch.
        
        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of dictionaries containing generated text and metadata.
        """
        return asyncio.run(self.generate_batch(prompts, **kwargs))

class LLMFactory:
    """
    Factory class for creating LLM interfaces.
    """
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register an LLM provider.
        
        Args:
            name: Name of the provider.
            provider_class: Class implementing the LLMInterface.
        """
        if not issubclass(provider_class, LLMInterface):
            raise TypeError(f"Provider class must implement LLMInterface: {provider_class}")
        
        cls._providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def create(cls, provider: str, model_id: Optional[str] = None, **kwargs) -> LLMInterface:
        """
        Create an LLM interface for the specified provider.
        
        Args:
            provider: Name of the provider.
            model_id: Identifier for the model to use.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            LLM interface instance.
            
        Raises:
            ValueError: If provider is not registered.
        """
        if provider not in cls._providers:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        # 处理 model_id 参数
        provider_kwargs = kwargs.copy()  # 创建 kwargs 的副本避免修改原字典
        
        if model_id is not None:
            # 如果提供了位置参数 model_id，且 kwargs 中没有同名参数，则添加到 kwargs
            if 'model_id' not in provider_kwargs:
                provider_kwargs['model_id'] = model_id
            # 如果两者都提供，使用位置参数 model_id
            else:
                logger.warning(f"Both positional and keyword model_id provided, using positional value: {model_id}")
                provider_kwargs['model_id'] = model_id
        elif 'model_id' not in provider_kwargs:
            # 如果两者都未提供，抛出错误
            raise ValueError("model_id must be provided either as positional or keyword argument")
        
        # 创建并返回提供程序实例
        return cls._providers[provider](**provider_kwargs)