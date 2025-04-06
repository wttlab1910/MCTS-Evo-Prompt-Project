"""
Module for expanding prompts with structured enhancements.
"""
from typing import Dict, Any, List, Optional
import os
import json
from app.utils.logger import get_logger
from app.config import PROMPT_GUIDE_DIR
from app.utils.cache import cached

logger = get_logger("input.prompt_expander")

class PromptExpander:
    """
    Expands prompts with structured enhancements based on Prompt Engineering Guide.
    """
    
    def __init__(self):
        """
        Initialize the prompt expander.
        """
        self.prompt_guide_dir = PROMPT_GUIDE_DIR
        self.prompt_guide_dir.mkdir(exist_ok=True, parents=True)
        
        # Load prompt patterns and templates
        self.patterns = self._load_patterns()
        self.templates = self._load_templates()
        
        logger.info(f"Prompt expander initialized with {len(self.patterns)} patterns and {len(self.templates)} templates")
    
    def expand(self, prompt: str, task_analysis: Dict[str, Any]) -> str:
        """
        Expand a prompt with structured enhancements.
        
        Args:
            prompt: Original prompt text.
            task_analysis: Analysis of the task.
            
        Returns:
            Expanded prompt.
        """
        # Select appropriate template based on task type
        template = self._select_template(task_analysis)
        
        # Apply structured expansion
        expanded_prompt = self._apply_structure(prompt, template, task_analysis)
        
        # Apply domain adaptation if applicable
        expanded_prompt = self._apply_domain_adaptation(expanded_prompt, task_analysis)
        
        # Enhance output format specification
        expanded_prompt = self._enhance_format(expanded_prompt, task_analysis)
        
        logger.debug(f"Expanded prompt: {expanded_prompt}")
        return expanded_prompt
    
    def _load_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load prompt patterns from the Prompt Engineering Guide.
        
        Returns:
            Dictionary of patterns by task type.
        """
        patterns_file = self.prompt_guide_dir / "patterns.json"
        
        # Create default patterns if file doesn't exist
        if not patterns_file.exists():
            self._create_default_patterns(patterns_file)
        
        # Load patterns
        try:
            with open(patterns_file, 'r') as f:
                patterns = json.load(f)
            return patterns
        except Exception as e:
            logger.error(f"Error loading prompt patterns: {e}")
            return self._create_default_patterns(patterns_file)
    
    def _create_default_patterns(self, patterns_file) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create default patterns file.
        
        Args:
            patterns_file: Path to patterns file.
            
        Returns:
            Dictionary of default patterns.
        """
        default_patterns = {
            "classification": [
                {"pattern": r"(?i)classify", "role": "Classification Expert", "steps": ["Analyze the provided content", "Identify key features", "Assign appropriate category"]},
                {"pattern": r"(?i)categorize", "role": "Categorization Specialist", "steps": ["Read the input carefully", "Identify distinguishing characteristics", "Determine the correct category"]}
            ],
            "extraction": [
                {"pattern": r"(?i)extract", "role": "Information Extraction Specialist", "steps": ["Read the source content", "Identify target information", "Extract relevant data"]},
                {"pattern": r"(?i)find", "role": "Information Finder", "steps": ["Scan the provided content", "Locate requested information", "Present extracted data"]}
            ],
            "summarization": [
                {"pattern": r"(?i)summarize", "role": "Content Summarizer", "steps": ["Read the full content", "Identify key points", "Create concise summary", "Verify completeness"]},
                {"pattern": r"(?i)key points", "role": "Key Points Extractor", "steps": ["Analyze the content", "Identify main ideas", "List key points", "Ensure coverage"]}
            ],
            "generation": [
                {"pattern": r"(?i)generate", "role": "Content Creator", "steps": ["Understand the requirements", "Develop content structure", "Generate comprehensive content", "Review for quality"]},
                {"pattern": r"(?i)write", "role": "Writer", "steps": ["Analyze the topic", "Create outline", "Write detailed content", "Edit for clarity"]}
            ],
            "question_answering": [
                {"pattern": r"(?i)answer", "role": "Question Answering Expert", "steps": ["Understand the question", "Analyze available information", "Formulate comprehensive answer", "Verify accuracy"]},
                {"pattern": r"(?i)why|how|what|when|where|who", "role": "Information Provider", "steps": ["Interpret the question", "Gather relevant information", "Provide clear, accurate answer"]}
            ]
        }
        
        # Save default patterns
        try:
            patterns_file.parent.mkdir(exist_ok=True, parents=True)
            with open(patterns_file, 'w') as f:
                json.dump(default_patterns, f, indent=2)
            logger.info(f"Created default prompt patterns at {patterns_file}")
        except Exception as e:
            logger.error(f"Error creating default prompt patterns: {e}")
        
        return default_patterns
    
    def _load_templates(self) -> Dict[str, str]:
        """
        Load prompt templates from the Prompt Engineering Guide.
        
        Returns:
            Dictionary of templates by task type.
        """
        templates_file = self.prompt_guide_dir / "templates.json"
        
        # Create default templates if file doesn't exist
        if not templates_file.exists():
            self._create_default_templates(templates_file)
        
        # Load templates
        try:
            with open(templates_file, 'r') as f:
                templates = json.load(f)
            return templates
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            return self._create_default_templates(templates_file)
    
    def _create_default_templates(self, templates_file) -> Dict[str, str]:
        """
        Create default templates file.
        
        Args:
            templates_file: Path to templates file.
            
        Returns:
            Dictionary of default templates.
        """
        default_templates = {
            "classification": """
Role: {role}
Task: Classify the following content into appropriate categories.
Steps:
{steps}
Content: {content}
Output: Provide the classification result with a brief explanation.
""".strip(),
            "extraction": """
Role: {role}
Task: Extract the requested information from the provided content.
Steps:
{steps}
Content: {content}
Output: Present the extracted information clearly.
""".strip(),
            "summarization": """
Role: {role}
Task: Summarize the following content effectively.
Steps:
{steps}
Content: {content}
Output: Provide a concise summary capturing the main points.
""".strip(),
            "generation": """
Role: {role}
Task: Generate content based on the given specifications.
Steps:
{steps}
Requirements: {content}
Output: Provide well-structured, comprehensive content that fulfills all requirements.
""".strip(),
            "question_answering": """
Role: {role}
Task: Answer the following question accurately and comprehensively.
Steps:
{steps}
Question: {content}
Output: Provide a clear, well-reasoned answer with relevant information.
""".strip(),
            "default": """
Role: {role}
Task: {content}
Steps:
{steps}
Output: Provide a well-structured, comprehensive response.
""".strip()
        }
        
        # Save default templates
        try:
            templates_file.parent.mkdir(exist_ok=True, parents=True)
            with open(templates_file, 'w') as f:
                json.dump(default_templates, f, indent=2)
            logger.info(f"Created default prompt templates at {templates_file}")
        except Exception as e:
            logger.error(f"Error creating default prompt templates: {e}")
        
        return default_templates
    
    def _select_template(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an appropriate template based on task analysis.
        
        Args:
            task_analysis: Analysis of the task.
            
        Returns:
            Template dictionary.
        """
        task_type = task_analysis["task_type"]
        
        # Find patterns matching this task type
        task_patterns = self.patterns.get(task_type, [])
        if not task_patterns:
            # Fall back to default if no patterns for this task type
            return {
                "role": "Task Specialist",
                "template": self.templates.get("default", self.templates["default"]),
                "steps": ["Understand the requirements", "Process the information", "Generate appropriate response"]
            }
        
        # Select the best matching pattern
        best_pattern = None
        for pattern in task_patterns:
            import re
            if re.search(pattern["pattern"], task_analysis.get("prompt", "")):
                best_pattern = pattern
                break
        
        # If no pattern matched, use the first one
        if not best_pattern and task_patterns:
            best_pattern = task_patterns[0]
        
        # Get template for this task type
        template_text = self.templates.get(task_type, self.templates["default"])
        
        return {
            "role": best_pattern["role"] if best_pattern else "Task Specialist",
            "template": template_text,
            "steps": best_pattern["steps"] if best_pattern else ["Analyze the input", "Process information", "Generate response"]
        }
    
    def _apply_structure(self, prompt: str, template: Dict[str, Any], task_analysis: Dict[str, Any]) -> str:
        """
        Apply structured expansion to the prompt.
        
        Args:
            prompt: Original prompt text.
            template: Selected template.
            task_analysis: Analysis of the task.
            
        Returns:
            Structured prompt.
        """
        # Format steps as bullet points
        steps_text = "\n".join([f"- {step}" for step in template["steps"]])
        
        # Apply template
        expanded = template["template"].format(
            role=template["role"],
            steps=steps_text,
            content=prompt
        )
        
        return expanded
    
    def _apply_domain_adaptation(self, prompt: str, task_analysis: Dict[str, Any]) -> str:
        """
        Apply domain-specific adaptations to the prompt.
        
        Args:
            prompt: Current prompt text.
            task_analysis: Analysis of the task.
            
        Returns:
            Domain-adapted prompt.
        """
        # Extract domain from key concepts if available
        domain = None
        for concept in task_analysis.get("key_concepts", []):
            # Check if concept matches a known domain
            domains = ["medical", "legal", "financial", "technical", "scientific", "educational"]
            for d in domains:
                if d in concept.lower():
                    domain = d
                    break
            if domain:
                break
        
        if not domain:
            return prompt  # No domain adaptation needed
        
        # Add domain-specific instructions
        domain_instructions = {
            "medical": "\nAdditional Instructions: Use appropriate medical terminology. Be precise and accurate with medical concepts. Consider patient privacy and ethical implications.",
            "legal": "\nAdditional Instructions: Use proper legal terminology. Be precise with legal concepts and definitions. Avoid making statements that could be construed as legal advice.",
            "financial": "\nAdditional Instructions: Use accurate financial terminology. Be precise with numbers and financial concepts. Include relevant disclaimers about financial information.",
            "technical": "\nAdditional Instructions: Use precise technical terminology. Include relevant technical details. Structure information for technical audiences.",
            "scientific": "\nAdditional Instructions: Follow scientific conventions. Be precise with scientific terminology. Base responses on established scientific knowledge.",
            "educational": "\nAdditional Instructions: Structure information for educational purposes. Use appropriate pedagogical approaches. Ensure content is accessible for the target learning level."
        }
        
        return prompt + domain_instructions.get(domain, "")
    
    def _enhance_format(self, prompt: str, task_analysis: Dict[str, Any]) -> str:
        """
        Enhance the output format specification.
        
        Args:
            prompt: Current prompt text.
            task_analysis: Analysis of the task.
            
        Returns:
            Format-enhanced prompt.
        """
        task_type = task_analysis["task_type"]
        
        # Add format specifications based on task type
        format_specifications = {
            "classification": "\nOutput Format: Provide the classification result in the format 'Category: [category name]' followed by a brief explanation.",
            "extraction": "\nOutput Format: List the extracted information as key-value pairs or in a structured format appropriate for the requested information.",
            "summarization": "\nOutput Format: Provide the summary in a concise paragraph. If appropriate, include bullet points for key takeaways.",
            "generation": "\nOutput Format: Structure the generated content with appropriate headings, paragraphs, and formatting for readability.",
            "question_answering": "\nOutput Format: Start with a direct answer to the question, followed by supporting details and explanations.",
            "sentiment_analysis": "\nOutput Format: State the sentiment clearly (e.g., 'Sentiment: Positive'), followed by a confidence level and key indicators.",
            "code_generation": "\nOutput Format: Provide code with appropriate comments. Include explanations for complex sections and instructions for usage.",
        }
        
        # Only add format specification if it's not already included
        if "Output Format:" not in prompt:
            prompt += format_specifications.get(task_type, "\nOutput Format: Provide a clear, well-structured response.")
        
        return prompt