"""
Module for separating prompts from data.
"""
import re
from typing import Tuple, Optional, List
from app.utils.logger import get_logger

logger = get_logger("input.prompt_separator")

class PromptSeparator:
    """
    Class for separating instruction prompts from data content.
    """
    
    def __init__(self):
        """
        Initialize the prompt separator.
        """
        # Common explicit delimiters
        self.explicit_delimiters = [
            # Format: (prefix_pattern, suffix_pattern)
            (r"(?i)instruction\s*:\s*", r"(?i)\s*data\s*:\s*"),
            (r"(?i)prompt\s*:\s*", r"(?i)\s*content\s*:\s*"),
            (r"(?i)system\s*:\s*", r"(?i)\s*user\s*:\s*"),
            (r"(?i)query\s*:\s*", r"(?i)\s*context\s*:\s*"),
        ]
        
    def separate(self, input_text: str) -> Tuple[str, str]:
        """
        Separate the input text into prompt and data components.
        
        Args:
            input_text: The complete input string.
            
        Returns:
            Tuple of (prompt, data).
        """
        # Try explicit delimiter separation
        result = self._try_explicit_delimiters(input_text)
        if result:
            logger.debug("Separated using explicit delimiters")
            return result
            
        # Try semantic boundary separation
        result = self._try_semantic_boundary(input_text)
        if result:
            logger.debug("Separated using semantic boundary")
            return result
            
        # Try structured format separation
        result = self._try_structured_format(input_text)
        if result:
            logger.debug("Separated using structured format")
            return result
            
        # Handle ambiguous case with best effort
        logger.debug("Using best effort separation")
        return self._handle_ambiguous(input_text)
    
    def _try_explicit_delimiters(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Try to separate using explicit delimiters.
        
        Args:
            text: Input text.
            
        Returns:
            Tuple of (prompt, data) if successful, None otherwise.
        """
        for prefix_pattern, suffix_pattern in self.explicit_delimiters:
            # 首先找到前缀模式，如"Instruction:"
            prefix_match = re.search(prefix_pattern, text)
            if prefix_match:
                # 找到前缀后，在剩余文本中查找后缀模式，如"Data:"
                remaining_text = text[prefix_match.end():]
                suffix_match = re.search(suffix_pattern, remaining_text)
                
                if suffix_match:
                    # 提取提示部分（前缀和后缀之间的文本）
                    prompt = remaining_text[:suffix_match.start()].strip()
                    
                    # 提取数据部分（后缀之后的所有文本）
                    data = remaining_text[suffix_match.end():].strip()
                    
                    return prompt, data
        
        return None
    
    def _try_semantic_boundary(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Try to separate using semantic boundaries.
        
        Args:
            text: Input text.
            
        Returns:
            Tuple of (prompt, data) if successful, None otherwise.
        """
        # 处理特定模式"Extract entities from: John..."
        exact_pattern = r'(Extract\s+entities\s+from:)\s+(.+)'
        exact_match = re.search(exact_pattern, text, re.IGNORECASE)
        if exact_match:
            prompt = exact_match.group(1).strip()
            data = exact_match.group(2).strip()
            return prompt, data
            
        # 首先尝试冒号作为分隔符（对于一般情况）
        colon_match = re.search(r'([^:]+):\s+(.+)', text)
        if colon_match:
            prompt = colon_match.group(1).strip()
            data = colon_match.group(2).strip()
            
            # 确保这是一个合理的分离 - 提示部分应该包含任务指令词汇
            task_indicators = ["extract", "find", "identify", "analyze", "summarize", "classify"]
            if any(indicator in prompt.lower() for indicator in task_indicators):
                return prompt, data
        
        # Common boundary phrases
        boundary_phrases = [
            r"\n\n",  # Double newline often separates instructions from data
            r"(?i)please\s+(?:analyze|process|evaluate|summarize|extract)",  # Task directives
            r"(?i)based\s+on\s+(?:the\s+)?(?:following|this)",  # Reference to content
            r"(?i)here(?:'s|\s+is)\s+(?:the|some)",  # Introduction of content
        ]
        
        for phrase in boundary_phrases:
            match = re.search(phrase, text)
            if match:
                # Split at the boundary
                if phrase == r"\n\n":
                    parts = text.split("\n\n", 1)
                    if len(parts) == 2:
                        return parts[0].strip(), parts[1].strip()
                else:
                    # For other phrases, consider the boundary part of the prompt
                    prompt = text[:match.end()].strip()
                    data = text[match.end():].strip()
                    
                    # Verify the split makes sense (both parts should be substantial)
                    if len(prompt) > 10 and len(data) > 10:
                        return prompt, data
        
        return None
    
    def _try_structured_format(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Try to separate based on structured formats like JSON.
        
        Args:
            text: Input text.
            
        Returns:
            Tuple of (prompt, data) if successful, None otherwise.
        """
        # Check for JSON-like format
        json_pattern = r'(\{[\s\S]*?"(?:instruction|prompt)"\s*:\s*"([^"]*)"[\s\S]*?"(?:data|content|input)"\s*:\s*"([^"]*)"[\s\S]*?\})'
        match = re.search(json_pattern, text)
        if match:
            try:
                prompt = match.group(2).strip()
                data = match.group(3).strip()
                return prompt, data
            except (IndexError, AttributeError):
                pass
        
        return None
    
    def _handle_ambiguous(self, text: str) -> Tuple[str, str]:
        """
        Handle ambiguous cases with best effort.
        
        Args:
            text: Input text.
            
        Returns:
            Tuple of (prompt, data).
        """
        # Heuristic approach: consider the first paragraph as the prompt
        # and the rest as data
        paragraphs = re.split(r'\n\s*\n', text)
        
        if len(paragraphs) == 1:
            # Single paragraph - try to split by sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) <= 2:
                # 对于非常短的文本，尝试使用问号分隔，通常问题是提示，回答是数据
                if "?" in text:
                    parts = text.split("?", 1)
                    return (parts[0] + "?").strip(), parts[1].strip()
                # 如果没有问号，将第一句作为提示，剩余部分作为数据
                if len(sentences) == 2:
                    return sentences[0].strip(), sentences[1].strip()
                # 如果只有一句，默认全为提示，无数据
                return text, ""
            else:
                # Consider first 1/3 of sentences as prompt
                split_point = max(1, len(sentences) // 3)
                prompt = ' '.join(sentences[:split_point])
                data = ' '.join(sentences[split_point:])
                return prompt, data
        else:
            # Multiple paragraphs
            # If the first paragraph has indicators of being a prompt, use it
            first_para = paragraphs[0]
            prompt_indicators = [
                r'(?i)analyze', r'(?i)explain', r'(?i)summarize', 
                r'(?i)find', r'(?i)extract', r'(?i)evaluate',
                r'(?i)please', r'(?i)can you'
            ]
            
            if any(re.search(pattern, first_para) for pattern in prompt_indicators):
                prompt = first_para
                data = '\n\n'.join(paragraphs[1:])
                return prompt, data
            else:
                # Otherwise, use a length-based heuristic
                # If first paragraph is short compared to the rest, it's likely a prompt
                if len(first_para) < 0.25 * len(text):
                    prompt = first_para
                    data = '\n\n'.join(paragraphs[1:])
                    return prompt, data
                else:
                    # Hard to determine, just split in half
                    half_point = len(text) // 2
                    prompt = text[:half_point].strip()
                    data = text[half_point:].strip()
                    return prompt, data