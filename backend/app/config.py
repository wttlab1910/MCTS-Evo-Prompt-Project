"""
Configuration management for MCTS-Evo-Prompt system.
Handles environment variables, system settings, and configuration parameters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
CACHE_DIR = DATA_DIR / "cached"
LOG_DIR = DATA_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, KNOWLEDGE_BASE_DIR, CACHE_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API settings
API_PREFIX = "/api/v1"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Authentication settings
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "False").lower() == "true"
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-7b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Optimization settings
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "50"))
EXPLORATION_CONSTANT = float(os.getenv("EXPLORATION_CONSTANT", "1.4142"))
MUTATION_RATE = float(os.getenv("MUTATION_RATE", "0.2"))
CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE", "0.2"))

# Cache settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
CACHE_EXPIRATION = int(os.getenv("CACHE_EXPIRATION", "3600"))  # In seconds

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "True").lower() == "true"
LOG_FILE = LOG_DIR / "mcts_evo_prompt.log"

# Prompt Engineering Guide settings - for training the prompt expansion model
PROMPT_GUIDE_DIR = KNOWLEDGE_BASE_DIR / "prompt_guide"
PROMPT_EXPANSION_MODEL_PATH = KNOWLEDGE_BASE_DIR / "models" / "prompt_expansion"