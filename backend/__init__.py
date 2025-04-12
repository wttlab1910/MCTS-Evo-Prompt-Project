"""
Backend package for MCTS-Evo-Prompt system.
"""
import sys
from pathlib import Path

# 添加backend目录到系统路径，确保其子模块可被正确导入
backend_path = Path(__file__).parent.resolve()
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))