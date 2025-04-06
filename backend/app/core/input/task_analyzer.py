"""
Module for analyzing and understanding tasks from prompts.
"""
from typing import Dict, Any, List, Tuple
import re
from app.utils.logger import get_logger

logger = get_logger("input.task_analyzer")

class TaskAnalyzer:
    """
    Analyzes prompts to understand the underlying task.
    """
    
    def __init__(self):
        """
        Initialize the task analyzer.
        """
        # Task type patterns
        self.task_patterns = {
            "classification": [
                r"(?i)\bclassify\b", r"(?i)\bcategorize\b", r"(?i)\bcategorise\b",
                r"(?i)identify the (type|category|class)",
                r"(?i)determine (the )?(type|category|class)",
                r"(?i)which (type|category|class)"
            ],
            "sentiment_analysis": [
                r"(?i)sentiment analysis", r"(?i)determine (the )?sentiment", 
                r"(?i)detect (the )?emotion", r"(?i)emotional tone",
                r"(?i)how does .* feel", r"(?i)positive or negative .* sentiment",
                r"(?i)if the sentiment is (positive|negative)"
            ],
            "extraction": [
                r"(?i)\bextract\b", r"(?i)\bfind\b", r"(?i)\blocate\b", 
                r"(?i)identify the", r"(?i)extract the main",
                r"(?i)what is the", r"(?i)who is", r"(?i)where is"
            ],
            "summarization": [
                r"(?i)\bsummarize\b", r"(?i)\bsummarise\b", r"(?i)\bsummary\b", 
                r"(?i)key points", r"(?i)main ideas", r"(?i)tldr"
            ],
            "generation": [
                r"(?i)generate", r"(?i)create", r"(?i)write", r"(?i)compose",
                r"(?i)draft", r"(?i)develop"
            ],
            "question_answering": [
                r"(?i)answer", r"(?i)why", r"(?i)how", r"(?i)what",
                r"(?i)when", r"(?i)where", r"(?i)who", r"(?i)which"
            ],
            "translation": [
                r"(?i)\btranslate\b", r"(?i)\bconvert\b", 
                r"(?i)from [a-z]+ to [a-z]+",
                r"(?i)in [a-z]+"
            ],
            "paraphrasing": [
                r"(?i)paraphrase", r"(?i)rephrase", r"(?i)rewrite",
                r"(?i)say .* differently", r"(?i)alternative way"
            ],
            "code_generation": [
                r"(?i)\bcode\b", r"(?i)\bprogram\b", r"(?i)\bfunction\b",
                r"(?i)\bimplement\b", r"(?i)\balgorithm\b", r"(?i)\bscript\b",
                r"(?i)write a (\w+ )?(function|program|code|script)",
                r"(?i)python function", r"(?i)javascript function"
            ],
        }
        
        # Map task types to evaluation methods
        self.evaluation_methods = {
            "classification": ["accuracy", "f1_score", "precision", "recall"],
            "extraction": ["exact_match", "f1_score", "partial_match"],
            "summarization": ["rouge", "bleu", "semantic_similarity"],
            "generation": ["perplexity", "human_evaluation", "semantic_similarity"],
            "question_answering": ["exact_match", "f1_score", "semantic_similarity"],
            "translation": ["bleu", "semantic_similarity"],
            "paraphrasing": ["semantic_similarity", "bleu"],
            "sentiment_analysis": ["accuracy", "f1_score"],
            "code_generation": ["functional_correctness", "code_quality", "execution_success"],
        }
        
        # Predefined categories mapping
        self.category_mapping = {
            "classification": ["text_classification", "multi_label_classification", "binary_classification"],
            "extraction": ["named_entity_recognition", "information_extraction", "keyword_extraction"],
            "summarization": ["text_summarization", "abstractive_summarization", "extractive_summarization"],
            "generation": ["text_generation", "story_generation", "essay_generation"],
            "question_answering": ["open_domain_qa", "closed_domain_qa", "factual_qa"],
            "translation": ["language_translation", "code_translation"],
            "paraphrasing": ["text_paraphrasing", "sentence_rephrasing"],
            "sentiment_analysis": ["sentiment_classification", "emotion_detection"],
            "code_generation": ["function_generation", "code_completion", "algorithm_implementation"],
        }
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze the prompt to understand the underlying task.
        
        Args:
            prompt: The prompt text.
            
        Returns:
            Dictionary with task analysis results.
        """
        # Identify task type
        task_type, confidence = self._identify_task_type(prompt)
        
        # Extract key concepts and entities
        concepts, entities = self._extract_concepts_entities(prompt)
        
        # Map to predefined categories
        category = self._map_to_category(task_type, prompt)
        
        # Determine appropriate evaluation methods
        evaluation_methods = self._determine_evaluation_methods(task_type)
        
        # Compile the analysis
        analysis = {
            "task_type": task_type,
            "task_confidence": confidence,
            "category": category,
            "key_concepts": concepts,
            "entities": entities,
            "evaluation_methods": evaluation_methods,
        }
        
        logger.debug(f"Task analysis: {analysis}")
        return analysis
    
    def _identify_task_type(self, text: str) -> Tuple[str, float]:
        """
        Identify the task type from the text.
        
        Args:
            text: Input text.
            
        Returns:
            Tuple of (task_type, confidence).
        """
        # Count matches for each task type
        task_scores = {}
        
        # 首先检查强指标模式（优先级顺序）
        
        # 代码生成检查 - 必须放在最前面，因为它可能同时包含"write"和"function"
        if re.search(r'(?i)(write|create|implement) a (\w+ )?(function|program|code|script|algorithm)', text) or re.search(r'(?i)(python|javascript|java|c\+\+|ruby) function', text):
            return "code_generation", 1.0
        
        # 分类检查
        if re.search(r'(?i)^(classify|categorize|categorise)\b', text):
            return "classification", 1.0
        
        # 摘要检查
        if re.search(r'(?i)^(summarize|summarise)\b', text):
            return "summarization", 1.0
            
        # 提取检查
        if re.search(r'(?i)^(extract|find)\b', text) or re.search(r'(?i)extract the (main )?(entities|information|keywords|names|dates)', text):
            return "extraction", 1.0
            
        # 情感分析检查
        if re.search(r'(?i)(determine|identify|analyze) (the )?sentiment', text) or re.search(r'(?i)sentiment (is|analysis)', text) or re.search(r'(?i)positive or negative', text):
            return "sentiment_analysis", 1.0
            
        # 翻译检查
        if re.search(r'(?i)^translate\b', text) or re.search(r'(?i)translate (this|the) .* (to|into) [a-z]+', text):
            return "translation", 1.0
            
        # 重述检查
        if re.search(r'(?i)^(paraphrase|rephrase|rewrite)\b', text):
            return "paraphrasing", 1.0
        
        # 检查所有任务类型
        for task_type, patterns in self.task_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text):
                    # 基础权重
                    weight = 1.0
                    
                    # 各种任务类型的特殊权重调整
                    if task_type == "sentiment_analysis" and re.search(r'(?i)\b(sentiment|positive|negative|emotion)\b', text):
                        weight = 1.5
                        
                    if task_type == "summarization" and re.search(r'(?i)\b(summarize|summarise|summary)\b', text):
                        weight = 1.5
                        
                    if task_type == "extraction" and re.search(r'(?i)\b(extract|find|locate)\b', text):
                        weight = 1.5
                        
                    if task_type == "code_generation" and re.search(r'(?i)\b(function|python|code|program)\b', text):
                        weight = 1.8
                        
                    if task_type == "question_answering" and text.strip().endswith("?"):
                        weight = 1.3
                        
                    matches += weight
            
            if matches > 0:
                # Normalize score based on the number of patterns
                task_scores[task_type] = matches / len(patterns)
        
        # 处理特殊冲突情况
        
        # 代码生成 vs 生成 - 当提到"function", "python", "code"等关键词时，优先选择代码生成
        if "code_generation" in task_scores and "generation" in task_scores:
            if re.search(r'(?i)\b(function|python|javascript|code|algorithm|program)\b', text):
                task_scores["code_generation"] += 0.5
        
        # 情感分析 vs 分类 - 当提到"sentiment"和"classify"时，需要根据上下文判断
        if "classification" in task_scores and "sentiment_analysis" in task_scores:
            if re.search(r'(?i)classify.*sentiment', text):
                # 分类情感，优先选择分类
                task_scores["classification"] = max(task_scores["classification"], task_scores["sentiment_analysis"] + 0.1)
            elif re.search(r'(?i)sentiment|positive or negative', text):
                # 直接提到情感，优先选择情感分析
                task_scores["sentiment_analysis"] = max(task_scores["sentiment_analysis"], task_scores["classification"] + 0.1)
        
        # 摘要 vs 翻译 - 当同时匹配时，根据关键词判断
        if "summarization" in task_scores and "translation" in task_scores:
            if re.search(r'(?i)\b(summarize|summarise|summary)\b', text):
                task_scores["summarization"] = max(task_scores["summarization"], task_scores["translation"] + 0.2)
        
        # 提取 vs 翻译 - 当同时匹配时，根据关键词判断
        if "extraction" in task_scores and "translation" in task_scores:
            if re.search(r'(?i)\b(extract|entities|extract the|identify the)\b', text):
                task_scores["extraction"] = max(task_scores["extraction"], task_scores["translation"] + 0.2)
        
        # If no task was identified, default to "generation"
        if not task_scores:
            return "generation", 0.5
        
        # Find the task with the highest score
        best_task = max(task_scores.items(), key=lambda x: x[1])
        return best_task[0], best_task[1]
    
    def _extract_concepts_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract key concepts and entities from the text.
        
        Args:
            text: Input text.
            
        Returns:
            Tuple of (concepts, entities).
        """
        # This is a simplified implementation
        # In a real system, you'd use NLP techniques like NER, keyword extraction, etc.
        
        # Simple extraction based on patterns
        concepts = []
        entities = []
        
        # Extract potential concepts (usually noun phrases)
        concept_patterns = [
            r"(?i)about\s+([a-z\s]+)",
            r"(?i)the\s+([a-z\s]+)\s+of",
            r"(?i)based on\s+([a-z\s]+)",
        ]
        
        for pattern in concept_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    concept = match.group(1).strip()
                    if len(concept) > 3 and concept not in concepts:  # Avoid too short concepts
                        concepts.append(concept)
                except (IndexError, AttributeError):
                    pass
        
        # Extract potential named entities (simplified)
        # Look for capitalized words that might be entities
        entity_matches = re.finditer(r'(?<!\.\s)(?<!\?\s)(?<!\!\s)([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)', text)
        for match in entity_matches:
            entity = match.group(1).strip()
            if len(entity) > 1 and entity not in entities:
                entities.append(entity)
        
        return concepts, entities
    
    def _map_to_category(self, task_type: str, text: str) -> str:
        """
        Map the task to a predefined category.
        
        Args:
            task_type: Identified task type.
            text: Original prompt text.
            
        Returns:
            Category string.
        """
        potential_categories = self.category_mapping.get(task_type, [])
        
        if not potential_categories:
            return task_type
        
        # Simple mapping based on keyword matching
        for category in potential_categories:
            # 修复: 使用正确的转义序列
            pattern = category.replace("_", "\\s+")
            if re.search(f"(?i){pattern}", text):
                return category
        
        # Default to the first category if no match
        return potential_categories[0]
    
    def _determine_evaluation_methods(self, task_type: str) -> List[str]:
        """
        Determine appropriate evaluation methods for the task.
        
        Args:
            task_type: Identified task type.
            
        Returns:
            List of evaluation method names.
        """
        return self.evaluation_methods.get(task_type, ["default_evaluation"])