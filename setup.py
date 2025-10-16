"""
RAGStrict - AI Context Enhancement Tool
一个轻量级的命令行工具，为AI助手提供代码和文档上下文
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# 读取requirements
requirements = [
    "typer[all]>=0.9.0",
    "rich>=13.7.0",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.19.0",
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "mcp>=0.9.0",
    "aiohttp>=3.8.0",  # For intranet API calls
]

setup(
    name="ragstrict",
    version="0.1.0",
    description="AI Context Enhancement Tool - 为AI助手提供代码和文档上下文",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/MOONL0323/RAGStrict",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "ragstrict": [
            "models/**/*",
            "config/**/*",
        ]
    },
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rags=ragstrict.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    license="MIT",
    keywords="rag ai context mcp code-analysis documentation",
)
