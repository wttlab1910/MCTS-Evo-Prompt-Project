"""
Asynchronous prompt evaluator using LLM for evaluation.
"""
from typing import Dict, Any, Optional
import asyncio
import time
import re
from app.core.mdp.state import PromptState
from app.llm.interface import LLMInterface
from app.utils.logger import get_logger

logger = get_logger("llm.evaluation.async_evaluator")

class AsyncPromptEvaluator:
    """
    Asynchronous evaluator for prompts using LLM.
    
    This class handles evaluation of prompts by querying an LLM
    to assess prompt quality for specific tasks.
    """
    
    def __init__(self, llm: LLMInterface, cache_size: int = 100):
        """
        Initialize an async prompt evaluator.
        
        Args:
            llm: LLM interface for evaluation.
            cache_size: Size of evaluation cache.
        """
        self.llm = llm
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.evaluation_count = 0
        logger.debug(f"Initialized AsyncPromptEvaluator with cache_size={cache_size}")
    
    async def initialize(self):
        """Initialize the evaluator."""
        # Nothing to initialize for now
        pass
    
    async def evaluate_prompt(self, state: PromptState, data: Optional[str] = None) -> float:
        """
        Evaluate a prompt state using the LLM.
        
        Args:
            state: Prompt state to evaluate.
            data: Sample data to use in evaluation (optional).
            
        Returns:
            Evaluation score between 0 and 1.
        """
        # Increment evaluation count
        self.evaluation_count += 1
        
        # Check cache first
        cache_key = f"{state.text}||{data or ''}"
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit ({self.cache_hits}/{self.evaluation_count})")
            return self.cache[cache_key]
        
        self.cache_misses += 1
        logger.debug(f"Cache miss ({self.cache_misses}/{self.evaluation_count})")
        
        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(state, data)
        
        try:
            # Query the LLM
            response = await self.llm.generate(evaluation_prompt)
            
            # Extract score from response
            text = response.get("text", "")
            score = self._extract_score(text)
            
            # Cache the result
            self._cache_result(cache_key, score)
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            return 0.5  # Default mid-range score on error
    
    def _build_evaluation_prompt(self, state: PromptState, data: Optional[str] = None) -> str:
        """
        Build an evaluation prompt for the LLM.
        
        Args:
            state: Prompt state to evaluate.
            data: Sample data to use in evaluation (optional).
            
        Returns:
            Evaluation prompt string.
        """
        prompt_text = state.text.strip()
        
        evaluation_prompt = f"""
Rate the quality of the following prompt on a scale from 0 to 10:

PROMPT:
{prompt_text}

Evaluate based on:
1. Clarity and precision
2. Structure and organization
3. Appropriate level of detail
4. Effectiveness for its purpose

Please analyze the prompt and provide a single numerical rating between 0 and 10, 
where 10 is excellent quality and 0 is poor quality.

Rating (0-10):
"""
        
        if data:
            evaluation_prompt = f"""
Rate the quality of the following prompt for processing this data on a scale from 0 to 10:

PROMPT:
{prompt_text}

DATA:
{data}

Evaluate based on:
1. Clarity and precision
2. Structure and organization
3. Appropriate level of detail
4. Effectiveness for its purpose
5. Relevance to the provided data

Please analyze the prompt and provide a single numerical rating between 0 and 10, 
where 10 is excellent quality and 0 is poor quality.

Rating (0-10):
"""
        
        return evaluation_prompt.strip()
    
    def _extract_score(self, text: str) -> float:
        """
        Extract a numerical score from LLM response text.
        
        Args:
            text: Response text from LLM.
            
        Returns:
            Normalized score between 0 and 1.
        """
        # Try to find a rating pattern like "Rating: 8" or "8/10"
        rating_patterns = [
            r"Rating(?:\s*\(0-10\))?:\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"^(\d+(?:\.\d+)?)$",
            r"score(?::|is)\s*(\d+(?:\.\d+)?)",
            r"rating(?::|is)\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*out of\s*10"
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-1 range
                    return min(1.0, max(0.0, score / 10.0))
                except ValueError:
                    continue
        
        # If no clear rating found, try to extract any number
        numbers = re.findall(r"(\d+(?:\.\d+)?)", text)
        for num in numbers:
            try:
                score = float(num)
                if 0 <= score <= 10:
                    return min(1.0, max(0.0, score / 10.0))
            except ValueError:
                continue
        
        # Default score if no rating found
        logger.warning(f"Could not extract score from text: {text[:100]}...")
        return 0.5
    
    def _cache_result(self, key: str, score: float):
        """
        Cache an evaluation result.
        
        Args:
            key: Cache key.
            score: Evaluation score.
        """
        # Implement simple LRU by removing oldest entry if cache is full
        if len(self.cache) >= self.cache_size:
            # Remove a random key to avoid complexity of true LRU
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = score