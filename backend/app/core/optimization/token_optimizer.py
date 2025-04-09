"""
Token optimization module for efficient prompt generation.
"""
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import re
import string
from app.core.mdp.state import PromptState
from app.utils.logger import get_logger

logger = get_logger("optimization.token_optimizer")

class TokenOptimizer:
    """
    Optimize prompts for token efficiency.
    
    This class implements various techniques for reducing token count
    while preserving information density and semantic meaning.
    """
    
    def __init__(self, aggressive_mode: bool = False):
        """
        Initialize a token optimizer.
        
        Args:
            aggressive_mode: Whether to use more aggressive optimization techniques.
        """
        self.aggressive_mode = aggressive_mode
        
        # Common filler words that can often be removed
        self.filler_words = {
            "actually", "basically", "certainly", "definitely", "essentially", 
            "generally", "honestly", "however", "indeed", "literally", 
            "obviously", "of course", "pretty much", "probably", "quite", 
            "rather", "really", "simply", "somewhat", "sort of", 
            "surely", "truly", "typically", "very", "virtually"
        }
        
        # Words to replace with shorter alternatives
        self.replacements = {
            "in order to": "to",
            "due to the fact that": "because",
            "on the basis of": "based on",
            "in the event that": "if",
            "under the circumstances that": "if",
            "in the process of": "while",
            "with regard to": "about",
            "with reference to": "about",
            "in relation to": "about",
            "concerning the matter of": "about",
            "it is important to note that": "",
            "it should be noted that": "",
            "it is worth noting that": "",
            "needless to say": "",
            "as a matter of fact": "",
            "for all intents and purposes": "",
            "at the present time": "now",
            "at this point in time": "now",
            "in spite of the fact that": "although",
            "because of the fact that": "because",
            "in the near future": "soon",
            "in a timely manner": "quickly",
        }
        
        # Initialize optimization stats
        self.reset_stats()
        
        logger.debug(f"Initialized TokenOptimizer with aggressive_mode={aggressive_mode}")
    
    def optimize(self, state: PromptState) -> Tuple[PromptState, Dict[str, Any]]:
        """
        Optimize a prompt state for token efficiency.
        
        Args:
            state: The prompt state to optimize.
            
        Returns:
            Tuple of (optimized_state, optimization_stats).
        """
        logger.debug(f"Optimizing prompt of length {len(state.text)}")
        
        # Reset statistics
        self.reset_stats()
        
        # Start with the original text
        text = state.text
        original_length = len(text)
        
        # Apply optimization techniques
        text = self._remove_filler_words(text)
        text = self._replace_verbose_phrases(text)
        text = self._consolidate_instructions(text)
        text = self._optimize_formatting(text)
        
        if self.aggressive_mode:
            text = self._shorten_sentences(text)
            text = self._convert_sentences_to_bullet_points(text)
            text = self._use_abbreviations(text)
        
        # Create a new state with optimized text
        optimized_state = PromptState(text)
        
        # Calculate statistics
        optimized_length = len(text)
        reduction = original_length - optimized_length
        reduction_percent = (reduction / original_length) * 100 if original_length > 0 else 0
        
        # Update overall statistics
        self.stats["original_length"] = original_length
        self.stats["optimized_length"] = optimized_length
        self.stats["characters_reduced"] = reduction
        self.stats["reduction_percent"] = reduction_percent
        
        logger.info(f"Token optimization complete: {reduction} characters reduced ({reduction_percent:.2f}%)")
        
        return optimized_state, self.stats
    
    def reset_stats(self) -> None:
        """Reset optimization statistics."""
        self.stats = {
            "original_length": 0,
            "optimized_length": 0,
            "characters_reduced": 0,
            "reduction_percent": 0,
            "filler_words_removed": 0,
            "phrases_replaced": 0,
            "instructions_consolidated": 0,
            "formatting_improvements": 0,
            "sentences_shortened": 0,
            "bullet_points_created": 0,
            "abbreviations_used": 0
        }
    
    def _remove_filler_words(self, text: str) -> str:
        """
        Remove unnecessary filler words from text.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        count = 0
        
        # Regular expression to match whole words from filler_words set
        pattern = r'\b(' + '|'.join(re.escape(word) for word in self.filler_words) + r')\b'
        
        # Count occurrences before removal
        count = len(re.findall(pattern, text, re.IGNORECASE))
        
        # Remove filler words
        processed_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up double spaces
        processed_text = re.sub(r' +', ' ', processed_text)
        
        self.stats["filler_words_removed"] = count
        return processed_text
    
    def _replace_verbose_phrases(self, text: str) -> str:
        """
        Replace verbose phrases with more concise alternatives.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        count = 0
        processed_text = text
        
        for verbose, concise in self.replacements.items():
            pattern = re.escape(verbose)
            matches = re.findall(pattern, processed_text, re.IGNORECASE)
            count += len(matches)
            processed_text = re.sub(pattern, concise, processed_text, flags=re.IGNORECASE)
        
        self.stats["phrases_replaced"] = count
        return processed_text
    
    def _consolidate_instructions(self, text: str) -> str:
        """
        Consolidate similar or redundant instructions.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        # Extract steps or instructions (lines starting with numbers or hyphens)
        lines = text.split('\n')
        steps = []
        non_steps = []
        count = 0
        
        # Find all instruction lines
        step_indices = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*[\d-]+\.?\s+', line) or line.strip().startswith('-'):
                step_indices.append(i)
        
        # Process consecutive steps
        if step_indices:
            i = 0
            while i < len(step_indices):
                # Find consecutive steps
                j = i
                while j + 1 < len(step_indices) and step_indices[j + 1] - step_indices[j] == 1:
                    j += 1
                
                # Process this group of consecutive steps
                if j > i:
                    # Got consecutive steps, consolidate them
                    group = [lines[step_indices[k]] for k in range(i, j + 1)]
                    consolidated = self._consolidate_step_group(group)
                    
                    # Replace the steps with consolidated ones
                    for k in range(i, j + 1):
                        idx = step_indices[k]
                        if k - i < len(consolidated):
                            lines[idx] = consolidated[k - i]
                        else:
                            lines[idx] = ''  # Remove consolidated step
                            count += 1
                
                i = j + 1
        
        # Remove empty lines
        processed_text = '\n'.join(line for line in lines if line.strip())
        
        self.stats["instructions_consolidated"] = count
        return processed_text
    
    def _consolidate_step_group(self, steps: List[str]) -> List[str]:
        """
        Consolidate a group of similar steps.
        
        Args:
            steps: List of step strings.
            
        Returns:
            Consolidated steps.
        """
        # This is a simplified implementation
        # A more sophisticated approach would use NLP techniques
        # to identify and merge semantically similar steps
        
        # For now, just look for very similar steps and combine them
        consolidated = []
        i = 0
        while i < len(steps):
            current = steps[i]
            j = i + 1
            similar_found = False
            
            while j < len(steps):
                # Simple similarity check based on shared words
                current_words = set(re.findall(r'\b\w+\b', current.lower()))
                next_words = set(re.findall(r'\b\w+\b', steps[j].lower()))
                
                common_words = current_words.intersection(next_words)
                if len(common_words) > 0.7 * min(len(current_words), len(next_words)):
                    # Steps are similar, combine them
                    current = self._combine_steps(current, steps[j])
                    similar_found = True
                    j += 1
                else:
                    break
            
            consolidated.append(current)
            i = j if similar_found else i + 1
        
        return consolidated
    
    def _combine_steps(self, step1: str, step2: str) -> str:
        """
        Combine two similar steps into one.
        
        Args:
            step1: First step.
            step2: Second step.
            
        Returns:
            Combined step.
        """
        # Extract the step number/prefix
        prefix_match = re.match(r'^\s*([\d-]+\.?\s+)', step1)
        prefix = prefix_match.group(1) if prefix_match else ""
        
        # Extract the content of both steps
        content1 = re.sub(r'^\s*[\d-]+\.?\s+', '', step1).strip()
        content2 = re.sub(r'^\s*[\d-]+\.?\s+', '', step2).strip()
        
        # Combine the content
        combined = f"{prefix}{content1}"
        
        # If they're not too similar, add the unique parts of content2
        if content2 not in content1 and content1 not in content2:
            combined += f" and {content2}"
        
        return combined
    
    def _optimize_formatting(self, text: str) -> str:
        """
        Optimize whitespace, linebreaks, and formatting.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text.
        """
        # Count initial newlines
        initial_newlines = len(re.findall(r'\n+', text))
        
        # Reduce multiple blank lines to a single blank line
        processed_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from each line
        lines = processed_text.split('\n')
        processed_text = '\n'.join(line.rstrip() for line in lines)
        
        # Count final newlines
        final_newlines = len(re.findall(r'\n+', processed_text))
        
        self.stats["formatting_improvements"] = initial_newlines - final_newlines
        return processed_text
    
    def _shorten_sentences(self, text: str) -> str:
        """
        Shorten long sentences by removing non-essential clauses.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text with shortened sentences.
        """
        if not self.aggressive_mode:
            return text
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        count = 0
        
        for i, sentence in enumerate(sentences):
            # Only process long sentences
            if len(sentence) > 100:
                # Remove non-essential clauses in parentheses
                shortened = re.sub(r'\s*\([^)]*\)\s*', ' ', sentence)
                
                # Remove clauses starting with "which", "that", etc.
                shortened = re.sub(r',\s*(which|that|who|whose|whom|where)\s+[^,;.!?]*', '', shortened)
                
                if len(shortened) < len(sentence):
                    sentences[i] = shortened
                    count += 1
        
        self.stats["sentences_shortened"] = count
        return ' '.join(sentences)
    
    def _convert_sentences_to_bullet_points(self, text: str) -> str:
        """
        Convert long sentences to bullet points where appropriate.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text with bullet points.
        """
        if not self.aggressive_mode:
            return text
        
        # Look for sentences with lists (e.g., "A, B, and C")
        lines = text.split('\n')
        count = 0
        
        for i, line in enumerate(lines):
            # Check if this line contains a list-like structure
            list_match = re.search(r'(:|\.) ([^,.]+)(, [^,.]+)+(, and|and) ([^,.]+)', line)
            if list_match:
                prefix = line[:list_match.start()]
                list_items = re.findall(r'(?:, )?((?:and )?[^,.:;]+)(?=[,.:;]|$)', line[list_match.start():])
                
                # Convert to bullet points
                bullet_list = [prefix]
                for item in list_items:
                    item = item.strip()
                    if item.startswith('and '):
                        item = item[4:]
                    if item:
                        bullet_list.append(f"- {item}")
                
                lines[i] = '\n'.join(bullet_list)
                count += 1
        
        self.stats["bullet_points_created"] = count
        return '\n'.join(lines)
    
    def _use_abbreviations(self, text: str) -> str:
        """
        Replace common phrases with abbreviations where appropriate.
        
        Args:
            text: Text to process.
            
        Returns:
            Processed text with abbreviations.
        """
        if not self.aggressive_mode:
            return text
        
        # Common abbreviations
        abbreviations = {
            "for example": "e.g.",
            "that is": "i.e.",
            "and so on": "etc.",
            "et cetera": "etc.",
            "versus": "vs.",
        }
        
        count = 0
        processed_text = text
        
        for phrase, abbr in abbreviations.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            matches = re.findall(pattern, processed_text, re.IGNORECASE)
            count += len(matches)
            processed_text = re.sub(pattern, abbr, processed_text, flags=re.IGNORECASE)
        
        self.stats["abbreviations_used"] = count
        return processed_text