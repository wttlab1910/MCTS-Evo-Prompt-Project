"""
Application initialization.
"""
import os
from app.config import create_directories

# Create necessary directories
create_directories()

# Set up version
__version__ = "0.1.0"