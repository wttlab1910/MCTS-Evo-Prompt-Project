"""
Tests unit initialization.
"""
import os
import sys

# 如果app模块未在sys.modules中，添加backend到路径
if 'app' not in sys.modules:
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)