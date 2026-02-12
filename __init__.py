"""
Entity Neutering - Text anonymization for preventing lookahead bias in LLMs

A methodology to pre-process text data for preventing lookahead bias in Large Language Models (LLMs),
particularly in financial text analysis.

Main Components:
- EntityNeutering: Core neutering pipeline with masking and paraphrasing
- CommonTools: Shared utilities for LLM processing and feature extraction

Example Usage:
    >>> from EntityNeutering import neuter_data, NEUTER_ARGS
    >>> import pandas as pd
    >>> df = pd.read_csv('your_data.csv')
    >>> neuter_data(df, **NEUTER_ARGS)

For more information, see the README.md file or visit:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182756
"""

from .EntityNeutering import neuter_data, neuter_wrapper, NEUTER_ARGS

__all__ = [
    'neuter_data',
    'neuter_wrapper',
    'NEUTER_ARGS',
]

__version__ = '0.1.0'
__author__ = 'Joseph Engelberg, Asaf Manela, William Mullins, Luka Vulicevic'
