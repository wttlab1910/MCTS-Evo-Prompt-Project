"""
Module for state transitions in the MDP framework.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import random
from app.utils.logger import get_logger
from app.core.mdp.state import PromptState
from app.core.mdp.action import Action

logger = get_logger("mdp.transition")

class StateTransition:
    """
    Handles transitions between states in the MDP framework.
    
    This class applies actions to states and manages any stochasticity 
    in the transition process.
    """
    
    def __init__(self, stochasticity: float = 0.0):
        """
        Initialize a state transition handler.
        
        Args:
            stochasticity: Level of randomness in transitions (0.0 to 1.0).
                0.0 means deterministic transitions.
                Values > 0.0 add increasing levels of random variations.
        """
        self.stochasticity = max(0.0, min(1.0, stochasticity))
        logger.debug(f"Initialized StateTransition with stochasticity: {self.stochasticity}")
    
    def apply(self, state: PromptState, action: Action) -> PromptState:
        """
        Apply an action to a state to generate a new state.
        
        Args:
            state: Current state.
            action: Action to apply.
            
        Returns:
            New state resulting from the action.
        """
        # Log the transition
        logger.debug(f"Applying action {action} to state {state}")
        
        # Check if action is applicable
        if not action.is_applicable(state):
            logger.warning(f"Action {action} is not applicable to state {state}")
            return state  # Return the original state unchanged
        
        # Apply the action to get the new state
        new_state = action.apply(state)
        
        # Apply stochasticity if enabled
        if self.stochasticity > 0:
            new_state = self._apply_stochasticity(new_state)
        
        logger.debug(f"Transition resulted in new state {new_state}")
        return new_state
    
    def _apply_stochasticity(self, state: PromptState) -> PromptState:
        """
        Apply random variations to a state based on stochasticity level.
        
        Args:
            state: State to modify.
            
        Returns:
            Modified state with random variations.
        """
        # If stochasticity is very low, just return the original state
        if random.random() > self.stochasticity:
            return state
        
        # Create a copy of the state
        new_state = state.copy()
        text = new_state.text
        
        # Apply random variations (more severe with higher stochasticity)
        variations = [
            self._reorder_paragraphs,
            self._simplify_sentences,
            self._add_filler_words,
            self._remove_random_words
        ]
        
        # Select a random variation based on stochasticity
        num_variations = max(1, int(self.stochasticity * len(variations)))
        selected_variations = random.sample(variations, num_variations)
        
        # Apply selected variations
        for variation in selected_variations:
            text = variation(text)
        
        # Create a new state with the modified text
        # Note: components are not automatically updated from the modified text
        # A real implementation might need to re-extract components or track changes
        result = PromptState(
            text=text,
            components=new_state.components,  # Keep original components
            metrics=new_state.metrics,
            history=new_state.history,
            parent=state,
            action_applied="stochastic_variation"
        )
        
        return result
    
    def _reorder_paragraphs(self, text: str) -> str:
        """Randomly reorder some paragraphs."""
        paragraphs = text.split("\n\n")
        if len(paragraphs) <= 2:
            return text  # Too few paragraphs to reorder
        
        # Don't reorder the first 1-2 paragraphs (usually important context/role)
        keep_intact = min(2, len(paragraphs) // 3)
        
        # Only shuffle a subset of middle paragraphs
        to_shuffle = paragraphs[keep_intact:-1] if len(paragraphs) > 3 else paragraphs[keep_intact:]
        if len(to_shuffle) <= 1:
            return text  # Too few paragraphs to shuffle
            
        # Shuffle the middle paragraphs
        shuffled = to_shuffle.copy()
        random.shuffle(shuffled)
        
        # Reconstruct the text
        result = paragraphs[:keep_intact] + shuffled
        if len(paragraphs) > 3:
            result += [paragraphs[-1]]  # Keep the last paragraph in place
            
        return "\n\n".join(result)
    
    def _simplify_sentences(self, text: str) -> str:
        """Simplify some sentences by truncating them."""
        sentences = []
        current_sentence = ""
        
        # Split into sentences (simple approach)
        for char in text:
            current_sentence += char
            if char in ".!?":
                sentences.append(current_sentence)
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence)
        
        # Simplify a few random sentences
        num_to_simplify = max(1, int(len(sentences) * self.stochasticity * 0.2))
        indices_to_simplify = random.sample(range(len(sentences)), min(num_to_simplify, len(sentences)))
        
        for idx in indices_to_simplify:
            sentence = sentences[idx]
            words = sentence.split()
            
            if len(words) <= 5:
                continue  # Sentence already short
                
            # Keep first part of the sentence
            simplified_length = random.randint(max(3, len(words) // 2), len(words) - 1)
            sentences[idx] = " ".join(words[:simplified_length]) + "."
        
        return "".join(sentences)
    
    def _add_filler_words(self, text: str) -> str:
        """Add some filler words to the text."""
        filler_words = [
            " basically",
            " essentially",
            " actually",
            " generally",
            " in general",
            " for the most part",
            " typically"
        ]
        
        sentences = []
        current_sentence = ""
        
        # Split into sentences (simple approach)
        for char in text:
            current_sentence += char
            if char in ".!?":
                sentences.append(current_sentence)
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence)
        
        # Add fillers to some sentences
        num_to_modify = max(1, int(len(sentences) * self.stochasticity * 0.15))
        if not sentences:  # Early return if no sentences
            return text
        
        indices_to_modify = random.sample(range(len(sentences)), min(num_to_modify, len(sentences)))
        
        for idx in indices_to_modify:
            filler = random.choice(filler_words)
            sentence = sentences[idx]
            
            # Fix: Check if sentence is not empty after stripping
            stripped = sentence.rstrip()
            if stripped and stripped[-1] in ".!?":
                # Insert before the final punctuation
                sentences[idx] = stripped[:-1] + filler + stripped[-1] + sentence[len(stripped):]
            else:
                # Just append the filler
                sentences[idx] = sentence + filler
        
        return "".join(sentences)
    
    def _remove_random_words(self, text: str) -> str:
        """Remove a few random words from the text."""
        words = text.split()
        if len(words) <= 10:
            return text  # Too few words to remove
            
        # Remove a few random words
        num_to_remove = max(1, int(len(words) * self.stochasticity * 0.05))
        indices_to_remove = random.sample(range(len(words)), min(num_to_remove, len(words)))
        
        # Don't remove words that might be structural
        words_to_keep = [
            "Role:", "Task:", "Steps:", "Output:", "Format:", "Example:", "Constraints:"
        ]
        
        indices_to_remove = [idx for idx in indices_to_remove 
                            if idx < len(words) - 1 and words[idx] not in words_to_keep]
        
        result = [word for idx, word in enumerate(words) if idx not in indices_to_remove]
        return " ".join(result)