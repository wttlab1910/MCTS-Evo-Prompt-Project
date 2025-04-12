"""
Setup script for MCTS-Evo-Prompt project.
"""
from setuptools import setup, find_packages

setup(
    name="mcts_evo_prompt",
    version="0.1.0",
    packages=find_packages(),
    description="A system for optimizing prompts using MCTS and Evolutionary algorithms",
    author="MCTS-Evo-Prompt Team",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "huggingface_hub>=0.10.0",
        "transformers>=4.12.0",
        "tqdm>=4.62.0",
        "aiohttp>=3.8.0",
        "nest_asyncio>=1.5.1",
        "fastapi>=0.78.0",
        "uvicorn>=0.17.6"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "mcts-prompt=backend.run_task:main",
        ],
    },
)