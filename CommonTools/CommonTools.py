
import time
import pandas as pd
from CommonTools.prompts import sentiment, de_neuter_name, mask_template, para_template, mask_template_iter, para_template_iter
from CommonTools.GPT_agent import GPT_Agent, Ollama_Agent
from CommonTools.credentials import GPT_API_KEY
from openai import OpenAI
from fuzzywuzzy import fuzz
import numpy as np
import re
import math
pd.options.mode.chained_assignment = None  # Disable chained assignment warning
COMMON_SUFFIXES = [
    'group', 'holdings', 'holding', 'ltd', 'corp', 'inc', 'co', 'llc', 'plc', 'gmbh', 'ag', 's.a.', 's.a', 'bv', 'nv',
    'sa', 'spa', 'pte', 'limited', 'incorporated', 'company', 'corporation', 'lp', 'llp', 'technologies', 'tech'
]
SUFFIX_PATTERN = re.compile(r'\b(?:' + '|'.join(COMMON_SUFFIXES) + r')\b', flags=re.IGNORECASE)
PARENS_PATTERN = re.compile(r'\([^)]*\)')  # Remove anything in parentheses
_DIR_RE = re.compile(
    r'(?i)direction\s*estimate\s*:\s*(.*?)\s*(?=,|$|\s+magnitude\s*estimate)'
)
_MAG_RE = re.compile(
    r'(?i)magnitude\s*estimate\s*:\s*(.*?)\s*(?=,|$)'
)
# Labels we might see after a value (used for lookaheads)
_NEXT_LABELS = r'(?:ticker|name|date|year|industry|sector)\s*estimate'
_PAT_TICKER   = re.compile(rf'(?i)ticker\s*estimate\s*:\s*(.*?)\s*(?:,|;)?\s*(?={_NEXT_LABELS}|$)')
_PAT_NAME     = re.compile(rf'(?i)name\s*estimate\s*:\s*(.*?)\s*(?:,|;)?\s*(?={_NEXT_LABELS}|$)')
_PAT_DATE     = re.compile(rf'(?i)(?:date|year)\s*estimate\s*:\s*(.*?)\s*(?:,|;)?\s*(?={_NEXT_LABELS}|$)')
_PAT_INDUSTRY = re.compile(rf'(?i)(?:industry|sector)\s*estimate\s*:\s*(.*?)\s*(?:,|;)?\s*(?={_NEXT_LABELS}|$)')


def StringInputWrapper(input_string, clean=False):
    """
    Prepares input strings for LLM processing by optionally removing problematic characters.

    Args:
        input_string: The text string to be processed
        clean: Boolean flag to determine if characters should be removed

    Returns:
        Cleaned string with problematic characters removed (if clean=True) or original string

    Note:
        Used as a pre-processing step before sending text to LLM API calls
    """
    if clean:
        characters_to_remove = ['\n', '\'', '*', '\\']
        for char in characters_to_remove:
            input_string = input_string.replace(char, '')
    return input_string

def StringOutputWrapper(output_string, clean=False, **kwargs):
    """
    Processes LLM response strings by optionally removing problematic characters.

    Args:
        output_string: The text string returned from an LLM
        clean: Boolean flag to determine if characters should be removed
        **kwargs: Additional keyword arguments (not used but allows for compatibility)

    Returns:
        Cleaned string with problematic characters removed (if clean=True) or original string

    Note:
        Used as a post-processing step after receiving text from LLM API calls
    """
    if clean:
        characters_to_remove = ['\n', '\'', '*', '\\']
        for char in characters_to_remove:
            output_string = output_string.replace(char, '')
    return output_string


def process_llm_responses(df, column_name, input_string, agent, provider='openai'):
    """
    Processes each row in a DataFrame by sending text to an LLM agent and storing responses.

    Args:
        df: DataFrame containing text data to process
        column_name: Name of the column to store LLM responses
        input_string: Name of the column containing text to send to the LLM
        agent: LLM agent instance (GPT_Agent or Ollama_Agent) with initialized prompt

    Returns:
        DataFrame with a new column containing LLM responses

    Note:
        Handles errors including token limits and unexpected API responses
        Logs processing time upon completion
    """
    df[column_name] = None
    start_time = time.time()
    for i, row in df.iterrows():
        try:
            response = agent.response(row[input_string], StringInputWrapper, StringOutputWrapper)
            if provider == 'openai':
                df.at[i, column_name] = response.choices[0].message.content.strip()
            elif provider == 'ollama':
                df.at[i, column_name] = response
        except Exception as e:
            error_message = str(e)
            if "context_length_exceeded" in error_message or "maximum context length" in error_message:
                print(f"\nToken limit error at row {i}, skipping.")
            elif "<!DOCTYPE html>" in error_message:
                print("Received an HTML error page instead of data.")
            else:
                print(f"\nUnexpected error at row {i}: {e}")
                df.at[i, column_name] = ""
            continue
    end_time = time.time()
    minutes = (end_time - start_time) / 60
    print(f"\nThis calculation took: {minutes:.2f} minutes")
    return df


def _make_agent(provider, gpt_api_key, system_prompt, model_name):
    """
    Returns an agent instance based on provider.
    provider: "openai" or "ollama"
    """
    if str(provider).lower() == 'ollama':
        # Uses your Ollama_Agent class (local model name like 'llama3', 'gemma:7b', etc.)
        return Ollama_Agent(system_prompt, model=model_name)
    # default to OpenAI GPT agent
    return GPT_Agent(OpenAI(api_key=gpt_api_key), system_prompt, model=model_name)


def process_llm_responses_dynamic(df, column_name, input_string, prompt_template=None,
                                  model=None, provider='openai', gpt_api_key=None):
    """
    Processes each row in a DataFrame using dynamically constructed prompts for LLMs.

    Args:
        df: DataFrame containing text data to process
        column_name: Name of the column to store LLM responses
        input_string: Name of the column containing text to send to the LLM
        prompt_template: Either a static string prompt or a dictionary with 'template' key
                         containing a list of components for dynamic prompt construction
        model: Optional model name (e.g., 'gpt-4' for OpenAI or 'llama3' for Ollama)
        provider: LLM provider - either 'openai' or 'ollama' (default: 'openai')
        gpt_api_key: API key for OpenAI (required if provider is 'openai')

    Returns:
        DataFrame with a new column containing LLM responses

    Note:
        Dynamic templates can include static text and references to DataFrame columns
        Creates a new agent for each row to use row-specific prompt data
        Handles errors and logs processing time
    """
    # Validate inputs
    if provider.lower() == 'openai' and gpt_api_key is None:
        # Try to get from global variable if it exists
        try:
            gpt_api_key = GPT_API_KEY
        except NameError:
            raise ValueError("gpt_api_key is required when provider is 'openai'")

    df[column_name] = None
    start_time = time.time()

    for i, row in df.iterrows():
        try:
            if isinstance(prompt_template, dict) and 'template' in prompt_template:
                # Dynamic template building
                prompt_parts = []
                for component in prompt_template['template']:
                    if isinstance(component, str):
                        # Static text component
                        prompt_parts.append(component)
                    elif isinstance(component, dict) and 'column' in component:
                        # Dynamic component from DataFrame column
                        column_name_ref = component['column']
                        if column_name_ref in df.columns:
                            # Get the value from this row
                            column_value = str(row[column_name_ref])

                            if column_value == '':
                                column_value = 'N/A'

                            # Add prefix if specified
                            if 'prefix' in component:
                                column_value = component['prefix'] + column_value

                            # Add suffix if specified
                            if 'suffix' in component:
                                column_value = column_value + component['suffix']

                            prompt_parts.append(column_value)
                        else:
                            print(f"Warning: Column '{column_name_ref}' not found in DataFrame")
                    else:
                        print(f"Warning: Invalid component in prompt template: {component}")

                full_prompt = ''.join(prompt_parts)
            else:
                # Static string prompt
                full_prompt = prompt_template

            # Create agent using the helper function
            current_agent = _make_agent(
                provider=provider,
                gpt_api_key=gpt_api_key,
                system_prompt=full_prompt,
                model_name=model
            )

            response = current_agent.response(row[input_string], StringInputWrapper, StringOutputWrapper)
            if provider == 'openai':
                df.at[i, column_name] = response.choices[0].message.content.strip()
            elif provider == 'ollama':
                df.at[i, column_name] = response

        except Exception as e:
            error_message = str(e)
            if "context_length_exceeded" in error_message or "maximum context length" in error_message:
                print(f"\nToken limit error at row {i}, skipping.")
            elif "<!DOCTYPE html>" in error_message:
                print("Received an HTML error page instead of data.")
            else:
                print(f"\nUnexpected error at row {i}: {e}")
                df.at[i, column_name] = ""

            continue

    end_time = time.time()
    minutes = (end_time - start_time) / 60
    print(f"\nThis calculation took: {minutes:.2f} minutes")
    return df


def extract_sentiment_features(df, response_column, new_columns):
    """
    Extracts sentiment direction and magnitude from LLM responses into new DataFrame columns.

    Args:
        df: DataFrame containing LLM sentiment analysis responses
        response_column: Name of column containing raw LLM sentiment responses
        new_columns: List of two column names [direction_column, magnitude_column] for extracted values

    Returns:
        DataFrame with two new columns containing extracted direction and magnitude values

    Note:
        Direction is typically 0 (bearish), 1 (bullish), or NA
        Magnitude is converted to numeric and ranges from 0-1
    """
    df[new_columns] = df[response_column].apply(lambda x: pd.Series(extract_direction_and_magnitude(x)))
    df[new_columns[1]] = pd.to_numeric(df[new_columns[1]], errors='coerce')
    return df

def extract_direction_and_magnitude(text):
    """
    Parses sentiment analysis text to extract direction and magnitude values.

    Args:
        text: String containing LLM sentiment response in the expected format
              with Direction Estimate and Magnitude Estimate fields

    Returns:
        Tuple of (direction, magnitude) where:
        - direction is '0' (bearish), '1' (bullish), or 'NA'
        - magnitude is a string representation of a float between 0-1 or 'NA'

    Note:
        Uses regex patterns (_DIR_RE, _MAG_RE) to find the values
        Handles various formats and normalizes 'NA'-type responses
        Gracefully handles None/NaN inputs
    """
    # handle None/NaN/non-strings gracefully
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 'NA', 'NA'
    s = str(text)

    # normalize: remove markdown ** and stray backticks/underscores
    s = s.replace('*', '').replace('`', '').replace('_', '')
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    d_match = _DIR_RE.search(s)
    m_match = _MAG_RE.search(s)

    direction = d_match.group(1).strip() if d_match else 'NA'
    magnitude = m_match.group(1).strip() if m_match else 'NA'

    # normalize common NA tokens
    def _norm_na(x):
        xl = x.lower()
        return 'NA' if xl in {'na', 'n/a', 'nan', 'none', 'missing'} else x

    direction = _norm_na(direction)
    magnitude = _norm_na(magnitude)

    return direction, magnitude

def _normalize_text(x):
    """
    Helper function to normalize text by removing markdown and standardizing whitespace.

    Args:
        x: Input text to normalize

    Returns:
        Normalized string with markdown characters removed and whitespace standardized

    Note:
        Used internally by extraction functions to prepare text for regex matching
    """
    # Remove markdown markers and collapse whitespace
    s = str(x).replace('*', '').replace('`', '').replace('_', '')
    return re.sub(r'\s+', ' ', s).strip()

def _get(pattern, s):
    """
    Helper function that extracts text matching a regex pattern.

    Args:
        pattern: Compiled regex pattern to search for
        s: String to search within

    Returns:
        Matched group with whitespace and trailing commas/semicolons removed, or 'NA' if no match
    """
    m = pattern.search(s)
    return m.group(1).strip().rstrip(',;') if m else 'NA'

def _norm_na(val):
    """
    Helper function to normalize various forms of 'not available' values.

    Args:
        val: Value to normalize

    Returns:
        'NA' if the value represents a missing/empty value, otherwise the trimmed string
    """
    if not isinstance(val, str):
        return 'NA'
    return 'NA' if val.strip().lower() in {'', 'na', 'n/a', 'nan', 'none', 'null', 'missing'} else val.strip()

def extract_it(text):
    """
    Extracts entity identification attempts (ticker, name, date, industry) from LLM output.

    Args:
        text: String containing LLM de-neutering response with company identification attempts

    Returns:
        Tuple of (ticker, name, date, industry) extracted from the text
        Each value is a string or 'NA' if not found

    Note:
        Uses regex patterns to find ticker, company name, date, and industry guesses
        Normalizes text and handles various NA formats
        Gracefully handles None/NaN inputs
    """
    # handle None/NaN/non-strings gracefully
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return 'NA', 'NA', 'NA', 'NA'
    s = _normalize_text(text)

    ticker   = _norm_na(_get(_PAT_TICKER, s))
    name     = _norm_na(_get(_PAT_NAME, s))
    date     = _norm_na(_get(_PAT_DATE, s))
    industry = _norm_na(_get(_PAT_INDUSTRY, s))

    return ticker, name, date, industry

def extract_it_features(df, response_column, new_columns):
    """
    Extracts entity identification features from LLM responses into DataFrame columns.

    Args:
        df: DataFrame containing LLM de-neutering responses
        response_column: Name of column containing raw LLM identification attempts
        new_columns: List of four column names for extracted values: 
                    [ticker_column, name_column, date_column, industry_column]

    Returns:
        DataFrame with new columns containing the extracted identification attempts

    Note:
        Applies extract_it function to each row and splits results into separate columns
    """
    df[new_columns] = df[response_column].apply(lambda x: pd.Series(extract_it(x)))
    return df


def clean_name(name: str) -> str:
    """
    Standardizes company names for comparison by removing common variations.

    Args:
        name: Company name string to be standardized

    Returns:
        Standardized string with lowercase letters, no parentheses, punctuation,
        common suffixes, or spaces

    Note:
        Uses predefined patterns (PARENS_PATTERN, SUFFIX_PATTERN) to remove
        common company name elements like 'Inc', 'Corp', etc.
    """
    name = name.lower()
    name = PARENS_PATTERN.sub('', name)  # Remove parentheticals like (THE)
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = SUFFIX_PATTERN.sub('', name)  # Remove common suffixes
    name = re.sub(r'\s+', '', name)      # Remove all spaces
    return name.strip()

def compare_names(row):
    """
    Calculates string similarity between actual company name and guessed name.

    Args:
        row: DataFrame row containing 'COMNAM' (actual company name) and
             'Name_Guess' (LLM's guessed company name) columns

    Returns:
        Integer similarity score from 0-100 (0 for missing values)

    Note:
        Uses Levenshtein distance ratio from fuzzywuzzy library
        Standardizes both strings using clean_name before comparison
    """
    if pd.isna(row['COMNAM']) or pd.isna(row['Name_Guess']):
        return 0

    name1 = clean_name(str(row['COMNAM']))
    name2 = clean_name(str(row['Name_Guess']))

    return fuzz.ratio(name1, name2)

def compare_names_generic(row, guess_column='Name_Guess'):
    """
    More flexible version of compare_names that works with any column name for guessed names.

    Args:
        row: DataFrame row containing 'COMNAM' (actual company name) column
        guess_column: Name of the column containing the guessed company name
                     (default: 'Name_Guess')

    Returns:
        Integer similarity score from 0-100 (0 for missing values)

    Note:
        Uses token_sort_ratio from fuzzywuzzy which is more robust for word order differences
        Standardizes both strings using clean_name before comparison
    """
    if pd.isna(row['COMNAM']) or pd.isna(row[guess_column]):
        return 0  # Handle missing values
    name1 = clean_name(str(row['COMNAM']).strip())
    name2 = clean_name(str(row[guess_column]).strip())
    similarity = fuzz.token_sort_ratio(name1, name2)
    return similarity

def check_if_identified(df, threshold=75, date_col='Date_Guess', name_col='Name_Guess', suffix='_neutered'):
    """
    Evaluates whether entity neutering was successful by determining if an LLM could identify
    the original entity from neutered text.

    Args:
        df: DataFrame containing original entity data and LLM guesses
        threshold: Similarity threshold (0-100) for considering a name guess correct (default: 75)
        date_col: Name of column containing date guesses (default: 'Date_Guess')
        name_col: Name of column containing company name guesses (default: 'Name_Guess')
        suffix: Suffix to add to output column names (default: '_neutered')

    Returns:
        DataFrame with additional columns:
        - date_guess_correct{suffix}: 1 if date guess is within 7 days of actual date, 0 otherwise
        - name_similarity{suffix}: Numeric similarity score between actual and guessed names
        - name_guess_correct{suffix}: 1 if name similarity exceeds threshold, 0 otherwise
        - iort_guess_correct{suffix}: 1 if either name OR date was correctly guessed, 0 otherwise

    Note:
        Correctly identified entities (iort_guess_correct{suffix}=1) indicate neutering failure
        Date comparison uses ±7 days tolerance
    """
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')

    # Check if date guess is correct (within 7 days)
    df[f'date_guess_correct{suffix}'] = np.where(
        (df['Date'].notna()) & (df[date_col].notna()) &
        (abs(df['Date'] - df[date_col]) <= pd.Timedelta(days=7)),
        1, 0
    )

    # Check name similarity
    df[f'name_similarity{suffix}'] = df.apply(lambda row: compare_names_generic(row, name_col), axis=1)
    df[f'name_guess_correct{suffix}'] = (df[f'name_similarity{suffix}'] >= threshold).astype(int)

    # Mark as identified if either name or date was correctly guessed
    df[f'iort_guess_correct{suffix}'] = np.where(
        (df[f'name_guess_correct{suffix}'] == 1) | (df[f'date_guess_correct{suffix}'] == 1),
        1, 0
    )

    return df


if __name__ == "__main__":
    print("Hello World")