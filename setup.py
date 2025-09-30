"""
Setup script for the Video Summarizer project.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="video-summarizer",
    version="2.0.0",
    author="AI Video Summarizer Team",
    author_email="contact@videosummarizer.ai",
    description="AI-Powered Video Summarizer with Advanced NLP and Meeting Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-summarizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "web": [
            "streamlit>=1.28.0",
            "plotly>=5.0.0",
            "pandas>=1.3.0",
        ],
        "nlp": [
            "spacy>=3.4.0",
            "transformers>=4.20.0",
            "torch>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-summarizer=main:main",
            "vs-cli=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords=[
        "video", "summarization", "ai", "nlp", "speech-recognition", 
        "meeting-analysis", "transcription", "machine-learning"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/video-summarizer/issues",
        "Source": "https://github.com/yourusername/video-summarizer",
        "Documentation": "https://github.com/yourusername/video-summarizer/wiki",
    },
)
