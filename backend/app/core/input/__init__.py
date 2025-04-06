"""
Input processing and initialization module.
"""
from app.core.input.prompt_separator import PromptSeparator
from app.core.input.task_analyzer import TaskAnalyzer
from app.core.input.prompt_expander import PromptExpander

# 条件导入，避免在没有torch时出错
try:
    from app.core.input.model_trainer import PromptModelTrainer
    __all__ = ["PromptSeparator", "TaskAnalyzer", "PromptExpander", "PromptModelTrainer"]
except ImportError:
    # 如果无法导入model_trainer (torch不可用)，则不导出它
    __all__ = ["PromptSeparator", "TaskAnalyzer", "PromptExpander"]