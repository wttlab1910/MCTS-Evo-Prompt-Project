"""
Mutation operations for evolutionary prompt optimization.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import random
import re
from app.core.mdp.state import PromptState
from app.utils.logger import get_logger

logger = get_logger("evolution.mutation")

class PromptMutator:
    """
    Implements mutation operations for prompt states.
    
    Mutations introduce random variations to prompts to help
    escape local optima and explore new areas of the prompt space.
    """
    
    def __init__(self, mutation_strength: float = 0.5):
        """
        Initialize a prompt mutator.
        
        Args:
            mutation_strength: Controls the intensity of mutations (0.0 to 1.0).
        """
        self.mutation_strength = min(1.0, max(0.0, mutation_strength))
        self.synonyms = self._load_synonyms()
        logger.debug(f"Initialized PromptMutator with mutation_strength={mutation_strength}")
    
    def mutate(self, state: PromptState) -> PromptState:
        """
        Apply a random mutation to a prompt state.
        
        Args:
            state: The prompt state to mutate.
            
        Returns:
            A new mutated prompt state.
        """
        # Select a mutation operator based on the state's characteristics
        mutation_operators = [
            self._vocabulary_substitution,
            self._structural_variation,
            self._content_reduction,
            self._content_expansion
        ]
        
        # Weight the operators based on what might be most beneficial
        weights = self._calculate_operator_weights(state)
        
        # Choose an operator
        operator = random.choices(mutation_operators, weights=weights, k=1)[0]
        
        # Apply the selected mutation
        mutated_text = operator(state.text)
        
        # 确保生成了不同的文本
        if mutated_text == state.text:
            # 如果没有变化，强制添加一些内容以确保变化
            tips = [
                "Note: Pay special attention to edge cases and exceptions.",
                "Important: Ensure all output is well-structured and easy to understand.",
                "Context: This task requires careful analysis and systematic approach.",
                "Tip: Consider alternative perspectives before finalizing your response."
            ]
            mutated_text = state.text + "\n\n" + random.choice(tips)
        
        # Create a new state with the mutated text
        mutated_state = PromptState(
            text=mutated_text,
            history=state.history + ["mutation"],
            parent=state,
            action_applied="mutation"
        )
        
        logger.debug(f"Applied {operator.__name__} mutation to state {state.state_id[:8]}")
        return mutated_state
    
    def _calculate_operator_weights(self, state: PromptState) -> List[float]:
        """
        Calculate weights for each mutation operator based on the state.
        
        Args:
            state: The prompt state to analyze.
            
        Returns:
            List of weights for each operator.
        """
        # Default equal weights
        weights = [1.0, 1.0, 1.0, 1.0]
        
        # Adjust based on state characteristics
        text = state.text
        text_length = len(text)
        
        # If text is very long, favor content reduction
        if text_length > 1000:
            weights[2] *= 2.0  # Increase weight for content_reduction
        
        # If text is very short, favor content expansion
        if text_length < 200:
            weights[3] *= 2.0  # Increase weight for content_expansion
        
        # If text lacks structure, favor structural variation
        if not re.search(r"(Role|Task|Steps|Output):", text):
            weights[1] *= 2.0  # Increase weight for structural_variation
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    def _vocabulary_substitution(self, text: str) -> str:
        """
        Replace some words with synonyms.
        
        Args:
            text: Original text.
            
        Returns:
            Text with some words replaced by synonyms.
        """
        words = text.split()
        
        # 计算要替换的词的数量
        num_to_replace = max(1, int(self.mutation_strength * len(words) * 0.1))  # 增加替换比例
        
        # 确保至少替换一个词
        if num_to_replace == 0:
            num_to_replace = 1
        
        # 随机选择要替换的词
        indices_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))
        
        # 跟踪是否实际进行了任何替换
        replacements_made = False
        
        for idx in indices_to_replace:
            word = words[idx]
            # 只替换字母单词，长度合理
            if word.isalpha() and len(word) > 3:
                # 检查是否有同义词
                lower_word = word.lower()
                if lower_word in self.synonyms and self.synonyms[lower_word]:
                    # 用随机同义词替换
                    synonym = random.choice(self.synonyms[lower_word])
                    
                    # 保留大小写
                    if word.isupper():
                        synonym = synonym.upper()
                    elif word[0].isupper():
                        synonym = synonym.capitalize()
                    
                    if synonym != word:  # 确保真的是不同的词
                        words[idx] = synonym
                        replacements_made = True
        
        # 如果没有找到可替换的词，强制替换一个通用词
        if not replacements_made and len(words) > 5:
            # 常见的可替换单词及其替换词
            common_replacements = {
                "analyze": "examine",
                "determine": "identify",
                "carefully": "thoroughly", 
                "provide": "give",
                "the": "this"
            }
            
            # 尝试查找这些常见词
            for i, word in enumerate(words):
                lower_word = word.lower()
                if lower_word in common_replacements:
                    replacement = common_replacements[lower_word]
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    replacements_made = True
                    break
                    
            # 如果还是没找到，在文本末尾添加一个随机提示
            if not replacements_made:
                tips = [
                    "Please consider all aspects carefully.",
                    "Make sure to be thorough in your analysis.",
                    "Remember to focus on the key elements."
                ]
                return " ".join(words) + "\n\n" + random.choice(tips)
        
        return " ".join(words)
    
    def _structural_variation(self, text: str) -> str:
        """
        Reorder sections of the prompt.
        
        Args:
            text: Original text.
            
        Returns:
            Text with reordered sections.
        """
        # Split into sections (assuming sections are separated by double newlines)
        sections = text.split("\n\n")
        
        if len(sections) <= 2:
            # 如果部分太少，添加一个新部分而不是重排
            tips = [
                "Note: Remember to analyze all relevant aspects.",
                "Tip: Carefully consider the context before responding.",
                "Important: Focus on accuracy and clarity in your response."
            ]
            return text + "\n\n" + random.choice(tips)
        
        # Identify key sections that should stay in place
        role_section = None
        task_section = None
        output_section = None
        
        for i, section in enumerate(sections):
            if re.search(r"^Role:", section, re.IGNORECASE):
                role_section = i
            elif re.search(r"^Task:", section, re.IGNORECASE):
                task_section = i
            elif re.search(r"^Output", section, re.IGNORECASE):
                output_section = i
        
        # Create list of movable sections
        movable_indices = [i for i in range(len(sections))
                          if i not in (role_section, task_section, output_section)]
        
        if len(movable_indices) < 2:
            # 如果没有足够的可移动部分，添加一个新部分
            tips = [
                "Context: This task requires careful consideration of details.",
                "Note: Pay attention to all relevant information.",
                "Reminder: Be precise in your analysis and response."
            ]
            return text + "\n\n" + random.choice(tips)
        
        # Shuffle movable sections
        to_shuffle = [sections[i] for i in movable_indices]
        random.shuffle(to_shuffle)
        
        # Reconstruct text with shuffled sections
        result = []
        for i in range(len(sections)):
            if i in movable_indices:
                result.append(to_shuffle.pop(0))
            else:
                result.append(sections[i])
        
        return "\n\n".join(result)
    
    def _content_reduction(self, text: str) -> str:
        """
        Remove some non-essential content from the prompt.
        
        Args:
            text: Original text.
            
        Returns:
            Text with some content removed.
        """
        # Split into sections
        sections = text.split("\n\n")
        
        # 如果部分太少，不要减少，而是做一个小修改
        if len(sections) <= 2:
            # 简化文本中的某些词
            simplified = text.replace("in order to", "to")
            simplified = simplified.replace("for the purpose of", "for")
            simplified = simplified.replace("take into consideration", "consider")
            
            # 确保有变化
            if simplified == text:
                return text + "\n\nKeep responses concise and to the point."
            return simplified
        
        # Identify essential sections (role, task, steps, output)
        essential_indices = []
        for i, section in enumerate(sections):
            if re.search(r"^(Role|Task|Steps|Output):", section, re.IGNORECASE):
                essential_indices.append(i)
        
        # Create list of non-essential sections
        non_essential = [i for i in range(len(sections)) if i not in essential_indices]
        
        # 如果没有非必要部分，做微小修改
        if not non_essential:
            # 尝试简化某些步骤或说明
            for i in range(len(sections)):
                if "steps:" in sections[i].lower():
                    steps = sections[i].split("\n")
                    if len(steps) > 3:  # 如果有多个步骤，删除一个
                        del steps[random.randint(1, len(steps)-1)]
                        sections[i] = "\n".join(steps)
                        return "\n\n".join(sections)
            
            # 如果没有找到可以简化的部分，添加简洁性提示
            return text + "\n\nFocus on essential information only."
        
        # Randomly remove some non-essential sections
        num_to_remove = max(1, int(self.mutation_strength * len(non_essential)))
        to_remove = random.sample(non_essential, min(num_to_remove, len(non_essential)))
        
        # Reconstruct text without removed sections
        result = [sections[i] for i in range(len(sections)) if i not in to_remove]
        
        return "\n\n".join(result)
    
    def _content_expansion(self, text: str) -> str:
        """
        Add new content to the prompt.
        
        Args:
            text: Original text.
            
        Returns:
            Text with additional content.
        """
        # Split into sections
        sections = text.split("\n\n")
        
        # Define possible expansions
        expansions = [
            "Note: Pay special attention to edge cases and exceptions.",
            "Important: Ensure all output is well-structured and easy to understand.",
            "Context: This task requires careful analysis and systematic approach.",
            "Tip: Consider alternative perspectives before finalizing your response.",
            "Guideline: Focus on providing accurate and relevant information."
        ]
        
        # Choose a random expansion
        expansion = random.choice(expansions)
        
        # Choose where to add the expansion (beginning, end, or middle)
        position = random.choice(["beginning", "end", "middle"])
        
        if position == "beginning":
            return expansion + "\n\n" + text
        elif position == "end":
            return text + "\n\n" + expansion
        else:  # middle
            if len(sections) <= 1:
                return text + "\n\n" + expansion
            
            # Insert at a random point in the middle
            insert_point = random.randint(1, len(sections) - 1)
            sections.insert(insert_point, expansion)
            return "\n\n".join(sections)
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """
        Load synonym dictionary for vocabulary substitution.
        
        Returns:
            Dictionary mapping words to their synonyms.
        """
        # In a real implementation, this would load from a file or database
        # For now, we'll use a small hardcoded dictionary
        return {
            "analyze": ["examine", "study", "investigate", "evaluate"],
            "identify": ["recognize", "detect", "pinpoint", "spot"],
            "determine": ["decide", "establish", "ascertain", "conclude"],
            "provide": ["supply", "deliver", "furnish", "offer"],
            "ensure": ["guarantee", "confirm", "verify", "make sure"],
            "important": ["crucial", "essential", "vital", "significant"],
            "carefully": ["thoroughly", "meticulously", "diligently", "attentively"],
            "consider": ["contemplate", "examine", "reflect on", "think about"],
            "clear": ["explicit", "obvious", "evident", "distinct"],
            "relevant": ["applicable", "pertinent", "appropriate", "germane"]
        }
    
    def set_mutation_strength(self, strength: float) -> None:
        """
        Set the mutation strength.
        
        Args:
            strength: New mutation strength (0.0 to 1.0).
        """
        self.mutation_strength = min(1.0, max(0.0, strength))
        logger.debug(f"Set mutation strength to {self.mutation_strength}")