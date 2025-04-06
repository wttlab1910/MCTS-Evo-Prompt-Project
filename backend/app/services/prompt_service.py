"""
Service for prompt processing and optimization.
"""
from typing import Dict, Any, Tuple
from app.core.input.prompt_separator import PromptSeparator
from app.core.input.task_analyzer import TaskAnalyzer
from app.core.input.prompt_expander import PromptExpander
from app.core.input.model_trainer import PromptModelTrainer
from app.utils.logger import get_logger
from app.utils.timer import timed
from app.config import PROMPT_EXPANSION_MODEL_PATH

logger = get_logger("services.prompt")

class PromptService:
    """
    Service for prompt-related operations.
    """
    
    def __init__(self):
        """
        Initialize the prompt service.
        """
        self.separator = PromptSeparator()
        self.analyzer = TaskAnalyzer()
        self.expander = PromptExpander()
        self.model_trainer = None  # Lazy initialization
        self.use_model = PROMPT_EXPANSION_MODEL_PATH.exists()
        
        logger.info("Prompt service initialized")
    
    @timed
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process input text to separate prompt and data, analyze task, and expand prompt.
        
        Args:
            input_text: Complete input text.
            
        Returns:
            Dictionary with processed information.
        """
        # Separate prompt and data
        prompt, data = self.separator.separate(input_text)
        
        # Analyze the task
        task_analysis = self.analyzer.analyze(prompt)
        
        # Add prompt to task analysis for future use
        task_analysis["prompt"] = prompt
        
        # Expand the prompt
        expanded_prompt = self.expand_prompt(prompt, task_analysis.get("task_type"))
        
        # Combine results
        result = {
            "original_input": input_text,
            "prompt": prompt,
            "data": data,
            "task_analysis": task_analysis,
            "expanded_prompt": expanded_prompt
        }
        
        return result
    
    @timed
    def expand_prompt(self, prompt: str, task_type: str = None) -> str:
        """
        Expand a prompt using model or rule-based approach.
        
        Args:
            prompt: Prompt text to expand.
            task_type: Optional task type override.
            
        Returns:
            Expanded prompt.
        """
        # Try model-based expansion if available
        if self.use_model:
            try:
                if self.model_trainer is None:
                    self.model_trainer = PromptModelTrainer()
                
                expanded_prompt = self.model_trainer.expand_prompt(prompt)
                logger.debug("Expanded prompt using trained model")
                return expanded_prompt
            except Exception as e:
                logger.warning(f"Model-based expansion failed, falling back to rule-based: {e}")
                self.use_model = False
        
        # Fallback to rule-based expansion
        # Analyze the task if task_type not provided
        if task_type is None:
            task_analysis = self.analyzer.analyze(prompt)
            task_type = task_analysis["task_type"]
        else:
            task_analysis = {"task_type": task_type, "prompt": prompt}
        
        # Add prompt to task analysis for future use
        task_analysis["prompt"] = prompt
        
        # Expand the prompt
        expanded_prompt = self.expander.expand(prompt, task_analysis)
        
        return expanded_prompt
    
    @timed
    def train_expansion_model(self, epochs=3, batch_size=8):
        """
        Train the prompt expansion model.
        
        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            
        Returns:
            Success status message.
        """
        try:
            if self.model_trainer is None:
                self.model_trainer = PromptModelTrainer()
            
            model_path = self.model_trainer.train(epochs=epochs, batch_size=batch_size)
            
            if model_path:
                self.use_model = True
                return {"status": "success", "message": f"Model trained and saved to {model_path}"}
            else:
                return {"status": "error", "message": "Model training failed"}
        except Exception as e:
            logger.error(f"Error training expansion model: {e}")
            return {"status": "error", "message": f"Error training model: {str(e)}"}