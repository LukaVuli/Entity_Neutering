"""
Setup script for Entity Neutering package
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # Package metadata
    name="entity-neutering",
    version="0.1.0",
    author="Joseph Engelberg, Asaf Manela, William Mullins, Luka Vulicevic",
    author_email="vulicevicluka22@gmail.com",
    description="Text anonymization for preventing lookahead bias in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukaVuli/Entity_Neutering",
    project_urls={
        "Bug Tracker": "https://github.com/LukaVuli/Entity_Neutering/issues",
        "Documentation": "https://github.com/LukaVuli/Entity_Neutering/blob/main/README.md",
        "Source Code": "https://github.com/LukaVuli/Entity_Neutering",
        "Paper": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182756",
    },

    # Package discovery
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),

    # Include non-Python files specified in MANIFEST.in
    include_package_data=True,

    # Dependencies
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "openai>=1.0.0",
        "requests>=2.25.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "nltk>=3.6.0",
        "gensim>=4.0.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "statsmodels>=0.12.0",
        "fuzzywuzzy>=0.18.0",
        "lxml>=4.6.0",
        "click>=8.0.0",
        "openpyxl>=3.0.0",
        "pillow>=8.0.0",
    ],

    # Optional dependencies
    extras_require={
        "ollama": ["ollama>=0.1.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
    },

    # Python version requirement
    python_requires=">=3.11",

    # Package classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Office/Business :: Financial",
        "Operating System :: OS Independent",
    ],

    # Keywords for PyPI search
    keywords=[
        "nlp",
        "llm",
        "text-processing",
        "anonymization",
        "financial-analysis",
        "sentiment-analysis",
        "bias-prevention",
        "entity-neutering",
        "text-anonymization",
        "lookahead-bias",
    ],

    # Entry points (optional - for command line tools)
    # Uncomment if you want to add CLI commands later
    # entry_points={
    #     'console_scripts': [
    #         'entity-neuter=EntityNeutering:main',
    #     ],
    # },

    # License
    license="MIT",

    # Ensure wheel is built as universal
    options={
        "bdist_wheel": {
            "universal": False,
        }
    },
)
