"""
Task adapter for MCTS-Evo-Prompt.
"""
import importlib
import sys
from pathlib import Path

# 确保tasks目录在Python路径中
tasks_dir = Path(__file__).resolve().parent
if str(tasks_dir) not in sys.path:
    sys.path.insert(0, str(tasks_dir))

# 确保项目根目录在路径中
project_root = tasks_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def get_task(task_name):
    """
    Get a task class by name.
    
    Args:
        task_name: Name of the task.
        
    Returns:
        Task class.
    """
    # 标准化任务名称为小写
    task_module_name = task_name.lower()
    
    # 尝试方法1: 使用绝对导入
    try:
        module = importlib.import_module(f"backend.tasks.{task_module_name}")
        return module.CustomTask
    except (ImportError, ModuleNotFoundError):
        pass
    
    # 尝试方法2: 作为直接子模块导入
    try:
        module = importlib.import_module(task_module_name)
        return module.CustomTask
    except (ImportError, ModuleNotFoundError):
        pass
    
    # 尝试方法3: 使用相对目录导入
    try:
        # 从当前包导入
        module = importlib.import_module(f".{task_module_name}", package="tasks")
        return module.CustomTask
    except (ImportError, ModuleNotFoundError) as e:
        # 最终错误
        raise ImportError(f"找不到任务 {task_name}，请确保任务模块存在并包含CustomTask类: {e}")