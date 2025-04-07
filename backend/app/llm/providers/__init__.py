"""
LLM provider implementations.
"""
# Import and register providers
from app.llm.providers.mistral import MistralProvider
from app.llm.providers.gemma import GemmaProvider
from app.llm.providers.ollama import OllamaProvider
from app.llm.interface import LLMFactory

# Register providers
LLMFactory.register_provider("mistral", MistralProvider)
LLMFactory.register_provider("gemma", GemmaProvider)
LLMFactory.register_provider("ollama", OllamaProvider)