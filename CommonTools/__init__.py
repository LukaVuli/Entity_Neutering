"""
CommonTools - Shared utilities for Entity Neutering

This package provides core functionality for LLM-based text processing,
including agent wrappers, response processing, and feature extraction.
"""

from .GPT_agent import GPT_Agent, Ollama_Agent
from .CommonTools import (
    _make_agent,
    process_llm_responses,
    process_llm_responses_dynamic,
    extract_sentiment_features,
    extract_it_features,
    check_if_identified
)
from .credentials import GPT_API_KEY
from .prompts import (
    sentiment,
    de_neuter_name,
    mask_template,
    para_template,
    mask_template_iter,
    para_template_iter
)

__all__ = [
    # Agents
    'GPT_Agent',
    'Ollama_Agent',
    # Core functions
    '_make_agent',
    'process_llm_responses',
    'process_llm_responses_dynamic',
    'extract_sentiment_features',
    'extract_it_features',
    'check_if_identified',
    # Credentials
    'GPT_API_KEY',
    # Prompts
    'sentiment',
    'de_neuter_name',
    'mask_template',
    'para_template',
    'mask_template_iter',
    'para_template_iter',
]

__version__ = '0.1.0'
