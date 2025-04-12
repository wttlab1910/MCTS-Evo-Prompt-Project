"""
Configuration management for the application.
"""

# llama3.3:latest    a6eb4748fd29    42 GB     
# Error: model requires more system memory (40.9 GiB) than is available (40.0 GiB)
# deepseek-r1:32b    38056bbcbb2d    19 GB     

# gemma3:12b         f4031aab637d    8.1 GB    

# mistral:latest     f974a74358d6    4.1 GB    
# llama3.1:latest    46e0c10c039e    4.9 GB    

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
# app/config.py
PROMPT_GUIDE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "prompt_guide"

# Knowledge base directories
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
DOMAIN_KNOWLEDGE_DIR = KNOWLEDGE_BASE_DIR / "domain_knowledge"
ERROR_PATTERNS_DIR = KNOWLEDGE_BASE_DIR / "error_patterns"
PROMPT_TEMPLATES_DIR = KNOWLEDGE_BASE_DIR / "prompt_templates"
PROMPT_GUIDE_DIR = KNOWLEDGE_BASE_DIR / "prompt_guide"

# Cache directories
CACHE_DIR = DATA_DIR / "cached"
PROMPTS_CACHE_DIR = CACHE_DIR / "prompts"
RESPONSES_CACHE_DIR = CACHE_DIR / "responses"
OPTIMIZATIONS_CACHE_DIR = CACHE_DIR / "optimizations"

# Log directory
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
PROMPT_EXPANSION_MODEL_PATH = MODELS_DIR / "prompt_expansion_model.pt"
# LLM configuration
LLM_CONFIG = {
    "default_provider": "ollama",  # Changed to ollama as default
    "providers": {
        # "mistral": {
        #     "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        #     "max_tokens": 2048,
        #     "temperature": 0.7,
        #     "timeout": 30,
        #     "local_path": None,  # Set path if using local model
        # },
        # "gemma": {
        #     "model_id": "google/gemma-7b-it",
        #     "max_tokens": 2048,
        #     "temperature": 0.7,
        #     "timeout": 30,
        #     "local_path": None,  # Set path if using local model
        # },
        "ollama": {
            "model_id": "mistral",  # Default Ollama model
            "api_base": "http://localhost:11434",
            "max_tokens": 2048,
            "temperature": 0.7,
            "timeout": 300
        },
        "ollama_gemma3": {
            "provider": "ollama",
            "model_id": "gemma3:12b",
            "api_base": "http://localhost:11434",
            "max_tokens": 2048,
            "temperature": 0.7,
            "timeout": 300
        },
        "ollama_deepseek": {
            "provider": "ollama",
            "model_id": "deepseek-r1:32b",
            "api_base": "http://localhost:11434",
            "max_tokens": 2048,
            "temperature": 0.7,
            "timeout": 300
        }
    },
    "batch_size": 5,
    "max_retries": 3,
    "retry_delay": 2,
    "cache_enabled": True
}

# API configuration
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": True,
    "reload": True,
    "workers": 1,
    "timeout": 60
}

# Create directory structure if it doesn't exist
def create_directories():
    """Create the application directory structure if it doesn't exist."""
    directories = [
        DATA_DIR,
        KNOWLEDGE_BASE_DIR,
        DOMAIN_KNOWLEDGE_DIR,
        ERROR_PATTERNS_DIR,
        PROMPT_TEMPLATES_DIR,
        PROMPT_GUIDE_DIR,
        PROMPT_GUIDE_DIR / "techniques",
        PROMPT_GUIDE_DIR / "templates",
        PROMPT_GUIDE_DIR / "examples",
        CACHE_DIR,
        PROMPTS_CACHE_DIR,
        RESPONSES_CACHE_DIR,
        OPTIMIZATIONS_CACHE_DIR,
        MODELS_DIR, 
        LOG_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Load environment-specific configuration
def load_env_config(env: str = None) -> Dict[str, Any]:
    """
    Load environment-specific configuration.
    
    Args:
        env: Environment name (development, production, etc.)
        
    Returns:
        Dictionary with configuration values.
    """
    if env is None:
        env = os.environ.get("APP_ENV", "development")
        
    config_file = BASE_DIR / f"config.{env}.json"
    
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return {}

# Override default configuration with environment-specific values
def update_config(config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update default configuration with environment-specific values.
    
    Args:
        config: Default configuration dictionary.
        env_config: Environment-specific configuration dictionary.
        
    Returns:
        Updated configuration dictionary.
    """
    for key, value in env_config.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
            
    return config

# Initialize configuration
ENV_CONFIG = load_env_config()
if "LLM_CONFIG" in ENV_CONFIG:
    LLM_CONFIG = update_config(LLM_CONFIG, ENV_CONFIG["LLM_CONFIG"])
if "API_CONFIG" in ENV_CONFIG:
    API_CONFIG = update_config(API_CONFIG, ENV_CONFIG["API_CONFIG"])

# Create directories on module import
create_directories()