# Entity Neutering

A methodology to pre-process text data for preventing lookahead bias in Large Language Models (LLMs), particularly in financial text analysis.

## Overview

Entity Neutering is a text preprocessing technique designed to anonymize financial documents and news articles to prevent LLMs from using prior knowledge about specific companies, industries, or time periods when making predictions. This ensures that model predictions are based solely on the textual content provided, eliminating lookahead bias that could artificially inflate model performance in text which is likely in the LLM's training sample.

## Authors

- [Joseph Engelberg](https://rady.ucsd.edu/faculty-research/faculty/joseph-engelberg.html)
- [Asaf Manela](https://asafmanela.github.io/)
- [William Mullins](https://willmullins.net/)
- [Luka Vulicevic](https://www.lukavulicevic.com/)

## Academic Paper

The full methodology and research findings are detailed in the academic paper available at:
**https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182756**

If you use this methodology in your research, please cite the original paper:
```
@article{engelberg2025entity,
  title={Entity Neutering},
  author={Engelberg, Joseph and Manela, Asaf and Mullins, William and Vulicevic, Luka},
  journal={Available at SSRN},
  year={2025}
}
```

## Key Features

### Text is masked and/or paraphrased iteratively until the same LLM cannot infer the entity.

1. **LLM-Based Masking** (Optional): Anonymization of all identifying features of the text
   - Company names → `Company_1`, `Company_2`, etc.
   - Industry terms → `industry_x`, `sector_x`
   - Dates and numbers → `time_x`, `number_x`
   - Products and services → `product_x`, `service_x`

2. **LLM-Based Paraphrasing** (Optional): Structural and lexical transformation to preserve anonymity
   - Maintains core information while making text unrecognizable
   - Changes sentence structure and vocabulary
   - Replaces identifying phrases and quotes

3. **Flexible Processing Modes**: Complete control over the neutering pipeline
   - Enable both masking and paraphrasing (default)
   - Use masking only
   - Use paraphrasing only
   - Skip neutering entirely (work with raw text)

4. **Iterative Refinement**: Up to 8 rounds (configurable) of additional masking and/or paraphrasing for texts that remain identifiable

5. **Built-in Identification Testing**: Automatic verification that neutering was successful
   - Tests if the same LLM can identify the entity after neutering
   - Tracks which round and step successfully neutered each text
   - Supports multiple LLM providers (OpenAI GPT models, Ollama local models)

6. **Robustness Testing**: Additional script for testing identification rates across different models
   - Measure identification success across various model sizes
   - Support for open-source models (Llama, Gemma) via Ollama

## Project Structure

```
Entity_Neutering/
├── README.md                 # This file
├── LICENSE                   # Project license
├── EntityNeutering.py        # Main processing pipeline
├── OllamaRobustness.py       # Robustness testing with Ollama models
├── CommonTools/              # Shared utilities and modules
│   ├── CommonTools.py        # Core processing functions
│   ├── GPT_agent.py          # OpenAI GPT agent wrapper
│   ├── prompts.py            # Template prompts for anonymization
│   └── credentials.py        # API credentials (not included)
```

## Neutering Parameters

The `NEUTER_ARGS` dictionary contains all configurable parameters for the entity neutering process. Here's a detailed breakdown of each parameter:

### Required Parameters

- **`output_dir`** (str): Directory where all output files will be saved
- **`gpt_api_key`** (str): Your API key (OpenAI API key if using OpenAI, not needed for Ollama)
- **`mask_template`** (template): Initial masking prompt template
- **`para_template`** (template): Initial paraphrasing prompt template
- **`mask_template_iter`** (template): Iterative masking prompt template for additional rounds
- **`para_template_iter`** (template): Iterative paraphrasing prompt template for additional rounds
- **`de_neuter_name`** (template): Template for testing text identification
- **`sentiment_template`** (template): Template for sentiment analysis

### Processing Control Parameters

- **`max_rounds`** (int, default: 8): Maximum number of iterative neutering rounds
- **`model_name`** (str, default: 'gemma3:4b'): Model to use for all operations (e.g., 'gpt-4o-mini' for OpenAI, 'llama3.2' or 'gemma3:4b' for Ollama)
- **`llm_provider`** (str, default: 'ollama'): LLM provider to use - either 'openai' or 'ollama'
- **`masking`** (bool, default: True): Enable masking step in the neutering pipeline
- **`paraphrase`** (bool, default: True): Enable paraphrasing step in the neutering pipeline
- **`extraction_name`** (str, default: 'LLM_RAW'): Column name for raw text sentiment extraction

### Output Control Parameters

- **`save_intermediate`** (bool, default: True): Save intermediate processing steps
- **`save_round_files`** (bool, default: True): Save separate files for each iterative round
- **`use_random_filename`** (bool, default: True): Generate random filenames to prevent collisions
- **`custom_filename_prefix`** (str, default: "instance"): Prefix for output filenames

### Sentiment Analysis Parameters

- **`perform_sentiment`** (bool, default: True): Enable sentiment analysis
- **`sentiment_on_raw`** (bool, default: True): Perform sentiment analysis on original text
- **`sentiment_on_neutered`** (bool, default: True): Perform sentiment analysis on neutered text

### Logging and Debugging Parameters

- **`verbose`** (bool, default: True): Print detailed progress messages
- **`log_errors`** (bool, default: True): Log errors to file
- **`error_log_path`** (str, default: None): Custom path for error log (uses output_dir if None)
- **`return_dataframe`** (bool, default: False): Return processed DataFrame instead of saving only

## Usage

### Complete Entity Neutering Process (Recommended)

For the full Entity Neutering process with default settings:

```python
import pandas as pd
from EntityNeutering import neuter_data, NEUTER_ARGS

# Load your financial text data
df = pd.read_csv('your_data.csv')

# Configure output directory
NEUTER_ARGS['output_dir'] = 'output_directory/'

# Run the complete entity neutering process with default settings
neuter_data(df, **NEUTER_ARGS)
```

> **Note**: For processing large datasets, run `EntityNeutering.py` directly after configuring paths in the script.
> The main script automatically uses multiprocessing to utilize all available CPU cores,
> significantly speeding up processing time.

### Customized Processing Variations

#### 1. Quick Processing (No Intermediate Files)

```python
from EntityNeutering import neuter_data, NEUTER_ARGS

# Minimal file saving for faster processing
custom_args = NEUTER_ARGS.copy()
custom_args.update({
    'save_intermediate': False,
    'save_round_files': False,
    'verbose': False,
})

neuter_data(df, **custom_args)
```

#### 2. Masking Only (No Paraphrasing)

```python
from EntityNeutering import neuter_data, NEUTER_ARGS

# Apply only masking without paraphrasing
masking_only_args = NEUTER_ARGS.copy()
masking_only_args.update({
    'masking': True,
    'paraphrase': False,
    'max_rounds': 8,
})

neuter_data(df, **masking_only_args)
```

#### 3. Paraphrasing Only (No Masking)

```python
from EntityNeutering import neuter_data, NEUTER_ARGS

# Apply only paraphrasing without masking
paraphrase_only_args = NEUTER_ARGS.copy()
paraphrase_only_args.update({
    'masking': False,
    'paraphrase': True,
    'max_rounds': 8,
})

neuter_data(df, **paraphrase_only_args)
```

#### 4. No Neutering (Sentiment Analysis on Raw Text Only)

```python
from EntityNeutering import neuter_data, NEUTER_ARGS

# Skip neutering entirely, only perform sentiment analysis
raw_text_only_args = NEUTER_ARGS.copy()
raw_text_only_args.update({
    'masking': False,
    'paraphrase': False,
    'perform_sentiment': True,
    'sentiment_on_raw': True,
    'sentiment_on_neutered': False,  # Can't perform sentiment on neutered if not neutering
})

neuter_data(df, **raw_text_only_args)
```

#### 5. Maximum Anonymization (High Security)

```python
from EntityNeutering import neuter_data, NEUTER_ARGS

# Maximum rounds with full logging
max_security_args = NEUTER_ARGS.copy()
max_security_args.update({
    'max_rounds': 12,  # More rounds for difficult-to-neuter texts
    'masking': True,
    'paraphrase': True,
    'save_intermediate': True,
    'save_round_files': True,
    'verbose': True,
    'log_errors': True
})

neuter_data(df, **max_security_args)
```

#### 6. Using OpenAI GPT Models

```python
from EntityNeutering import neuter_data, NEUTER_ARGS

# Use OpenAI GPT models instead of Ollama
openai_args = NEUTER_ARGS.copy()
openai_args.update({
    'llm_provider': 'openai',
    'model_name': 'gpt-4o-mini',  # or 'gpt-4-turbo', 'gpt-4', etc.
    'gpt_api_key': 'your-openai-api-key',
    'max_rounds': 6,
})

neuter_data(df, **openai_args)
```

#### 7. Using Different Ollama Models

```python
from EntityNeutering import neuter_data, NEUTER_ARGS

# Use different Ollama models
ollama_args = NEUTER_ARGS.copy()
ollama_args.update({
    'llm_provider': 'ollama',
    'model_name': 'llama3.2',  # or 'gemma3:27b', 'mistral', etc.
    'max_rounds': 8,
})

neuter_data(df, **ollama_args)
```

## Installation & Requirements

### Python Environment
- **Python Version**: 3.11.11
- **Package Manager**: condavenv

### Dependencies
```python
# Core packages
pandas
numpy
matplotlib
seaborn
plotly

# NLP and ML
nltk
gensim
scikit-learn
scipy
statsmodels

# LLM Integration
openai
ollama
requests

# Text Processing
fuzzywuzzy
lxml

# Utilities
click
openpyxl
pillow
```


## Data Requirements

Your input DataFrame must contain these required columns to use the built in prompts:

- **`Body Text`**: The text content to be neutered
- **`COMNAM`**: Company name
- **`Companies`**: Company ticker symbol  
- **`SIC_Industry`**: Industry classification
- **`Date`**: Date information (will be converted to datetime)

## Output Files

The neutering process generates several output files depending on your configuration:

### Per-Instance Files
- **`*_step1_masked.csv`**: After initial masking (if `save_intermediate=True` and `masking=True`)
- **`*_step2_paraphrased.csv`**: After initial paraphrasing (if `save_intermediate=True` and `paraphrase=True`)
- **`*_masking_step_X_neutered.csv`**: Successfully neutered texts from masking in round X (if `save_round_files=True`)
- **`*_paraphrasing_step_X_neutered.csv`**: Successfully neutered texts from paraphrasing in round X (if `save_round_files=True`)
- **`*_round_X_masked.csv`**: Intermediate masked text in round X (if `save_intermediate=True` and `save_round_files=True`)
- **`*_round_X_paraphrased.csv`**: Intermediate paraphrased text in round X (if `save_intermediate=True` and `save_round_files=True`)
- **`*_NEUTERED.csv`**: All neutered texts before sentiment analysis
- **`*_sent_neut.csv`**: After sentiment analysis on neutered text (if `save_intermediate=True` and `sentiment_on_neutered=True`)
- **`*_FINAL_FINAL_FINAL.csv`**: Final output with all processing complete

### Combined Output
- **`Finished_Data.csv`**: Combined results from all parallel processes (generated by main script)

### Key Output Columns
- **`Body Text Neutered`**: The final neutered text
- **`neutered_at_round`**: Which iteration round successfully neutered the text (0 for initial round, -1 for never neutered)
- **`neutered_at_step`**: Which step successfully neutered the text (e.g., 'masking_step_0', 'paraphrasing_step_2', 'never_neutered', 'no_neutering')
- **`iort_guess_correct_neutered`**: Binary indicator (0/1) of whether the LLM successfully identified the entity after neutering (0 = success, 1 = failure)
- **`Direction`** / **`Direction_neut`**: Sentiment direction on raw/neutered text (if sentiment analysis enabled)
- **`Magnitude`** / **`Magnitude_neut`**: Sentiment magnitude on raw/neutered text (if sentiment analysis enabled)

## Research Applications

Entity Neutering is particularly valuable for:

1. **Financial NLP Research**: Eliminating lookahead bias in financial sentiment analysis and text-based predictions
2. **Bias-Free Model Evaluation**: Testing LLM capabilities without the confounding effect of memorized information
3. **Privacy Protection**: Anonymizing sensitive financial documents and news articles
4. **Sentiment Analysis**: Studying firm-level news sentiment independent of company identity
5. **Robustness Testing**: Evaluating whether text-based signals are truly content-driven or identity-driven

## Robustness Testing Across Models

The `OllamaRobustness.py` script allows you to test how well your neutered texts perform across different LLM models. This helps verify that neutering is robust across various model architectures and sizes.

### Usage

```python
from OllamaRobustness import process_llm_responses
import pandas as pd

# Load neutered data
df = pd.read_csv('neutered_data.csv')

# Test with Llama 3.2
df = process_llm_responses(
    df,
    'LLM_DENEUTER_GUESS_LLAMA_3B',
    'Body Text Neutered',
    de_neuter_name,
    model_to_use='llama3.2'
)

# Test with Gemma
df = process_llm_responses(
    df,
    'LLM_DENEUTER_GUESS_GEMMA_4B',
    'Body Text Neutered',
    de_neuter_name,
    model_to_use='gemma3:4b'
)
```

This allows you to measure identification rates across different models to ensure your neutering approach generalizes well.

## Performance Considerations

- **Multiprocessing**: The main script automatically utilizes all CPU cores for parallel processing of large datasets
- **Flexible LLM Providers**:
  - **Ollama** (default): Free local models with no API costs, ideal for large-scale processing
  - **OpenAI**: Cloud-based models with API costs but potentially higher quality
- **API Rate Limiting**: Built-in error handling for API rate limits and timeout issues
- **Incremental Saving**: Regular saving of intermediate results to prevent data loss
- **Memory Management**: Efficient processing of large datasets through chunking and parallel execution

## Contributing

This codebase supports ongoing research into lookahead bias prevention in financial NLP. Contributions are welcome, particularly in the following areas:
- Additional anonymization strategies beyond masking and paraphrasing
- New evaluation metrics for measuring neutering effectiveness
- Integration with additional LLM providers (e.g., Anthropic Claude, Google Gemini API)
- Performance optimizations for processing speed
- Enhanced prompt templates for specific use cases
- Support for additional languages beyond English

## License

See `LICENSE` file for project licensing terms.
