"""
Prompt service for handling prompt operations.
"""
from typing import Dict, Any, List, Optional, Union
import asyncio
from app.core.input.prompt_separator import PromptSeparator
from app.core.input.task_analyzer import TaskAnalyzer
from app.core.input.prompt_expander import PromptExpander
from app.llm.interface import LLMFactory
from app.llm.evaluation.validator import ResponseValidator
from app.config import LLM_CONFIG
from app.utils.logger import get_logger
from app.utils.timer import Timer

logger = get_logger("services.prompt_service")

class PromptService:
    """
    Service for prompt operations.
    
    This service handles prompt separation, analysis, expansion, and evaluation.
    """
    
    def __init__(self):
        """Initialize the prompt service."""
        self.separator = PromptSeparator()
        self.analyzer = TaskAnalyzer()
        self.expander = PromptExpander()
        
        # Initialize LLM interface
        self._init_llm()
        
        logger.info("Prompt service initialized.")
    
    def _init_llm(self):
        """Initialize LLM interface."""
        provider = LLM_CONFIG["default_provider"]
        provider_config = LLM_CONFIG["providers"][provider]
        
        try:
            self.llm = LLMFactory.create(
                provider=provider,
                model_id=provider_config["model_id"],
                **provider_config
            )
            
            logger.info(f"Using {provider} LLM: {provider_config['model_id']}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process input text to separate prompt and data, analyze task, and expand prompt.
        
        Args:
            input_text: Complete input text.
            
        Returns:
            Dictionary with processed information.
        """
        with Timer("prompt_service.process_input", log_level="debug"):
            # Separate prompt and data
            prompt, data = self.separator.separate(input_text)
            
            # Analyze the task
            task_analysis = self.analyzer.analyze(prompt)
            task_analysis["prompt"] = prompt
            
            # Expand the prompt
            expanded_prompt = self.expand_prompt(prompt, task_analysis.get("task_type"))
            
            # Combine results
            return {
                "original_input": input_text,
                "prompt": prompt,
                "data": data,
                "task_analysis": task_analysis,
                "expanded_prompt": expanded_prompt
            }
    
    def expand_prompt(self, prompt: str, task_type: Optional[str] = None) -> str:
        """
        Expand a prompt using prompt engineering techniques.
        
        Args:
            prompt: Input prompt.
            task_type: Type of task (optional).
            
        Returns:
            Expanded prompt.
        """
        with Timer("prompt_service.expand_prompt", log_level="debug"):
            # Create task analysis if not available
            task_analysis = None
            if task_type:
                task_analysis = {
                    "task_type": task_type,
                    "task_confidence": 1.0,
                    "prompt": prompt
                }
            else:
                task_analysis = self.analyzer.analyze(prompt)
                task_analysis["prompt"] = prompt
            
            # Expand the prompt
            return self.expander.expand(prompt, task_analysis)
    
    async def evaluate_prompt(self, 
                            prompt: str, 
                            task_type: Optional[str] = None,
                            data: Optional[str] = None,
                            expected_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a prompt's quality.
        
        Args:
            prompt: Input prompt.
            task_type: Type of task (optional).
            data: Input data (optional).
            expected_output: Expected output (optional).
            
        Returns:
            Evaluation results.
        """
        with Timer("prompt_service.evaluate_prompt", log_level="debug"):
            # Ensure LLM is initialized
            if not self.llm:
                self._init_llm()
                if not self.llm:
                    raise RuntimeError("LLM interface could not be initialized")
            
            # Create complete input with data if provided
            complete_input = prompt
            if data:
                complete_input = f"{prompt}\n\n{data}"
            
            # Generate response from LLM
            response = await self.llm.generate(complete_input)
            
            # Create validator
            validator = ResponseValidator(task_type)
            
            # Validate response
            validation = validator.validate(response, expected_output)
            
            # Add additional metrics
            metrics = {
                "prompt_length": len(prompt),
                "data_length": len(data) if data else 0,
                "response_length": len(response.get("text", "")),
                "elapsed_time": response.get("elapsed_time", 0)
            }
            
            # Combine results
            return {
                "prompt": prompt,
                "task_type": task_type,
                "response": response.get("text", ""),
                "validation": validation,
                "metrics": metrics
            }
    
    def evaluate_prompt_sync(self, 
                           prompt: str, 
                           task_type: Optional[str] = None,
                           data: Optional[str] = None,
                           expected_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous version of evaluate_prompt.
        
        Args:
            prompt: Input prompt.
            task_type: Type of task (optional).
            data: Input data (optional).
            expected_output: Expected output (optional).
            
        Returns:
            Evaluation results.
        """
        return asyncio.run(self.evaluate_prompt(prompt, task_type, data, expected_output))