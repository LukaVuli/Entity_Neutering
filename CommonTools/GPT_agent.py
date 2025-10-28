import pandas as pd
from CommonTools.credentials import GPT_API_KEY #replace this with your own OpenAI API KEY
from openai import OpenAI
import re
import time
from collections import deque
import ollama
from abc import ABC, abstractmethod
import requests
from requests.auth import HTTPBasicAuth
import os
time_token_tracker = deque()


def remove_old_tokens(time_token_tracker):
    """
    Removes tokens from the tracker that are older than 60 seconds.

    Args:
        time_token_tracker: A deque containing tuples of (timestamp, token_count)

    Returns:
        None - modifies the deque in-place

    Note:
        Used for rate limiting API calls based on a rolling window of token usage
        Helps prevent exceeding API rate limits by tracking recent token consumption
    """
    current_time = time.time()
    while time_token_tracker and current_time - time_token_tracker[0][0] > 60:
        time_token_tracker.popleft()

def estimate_token_count(word_count):
    """
    Estimates the number of tokens based on word count.

    Args:
        word_count: Integer count of words in a text

    Returns:
        Integer estimate of tokens (using 1.45 tokens per word as a heuristic)

    Note:
        This is a rough approximation as actual tokenization varies by model
        The 1.45 multiplier is based on empirical observations of English text
    """
    token_estimate = word_count * 1.45
    return int(token_estimate)

class LLM_Agent(ABC):
    """
    Abstract base class for LLM agents (language model interfaces).

    This class defines the common interface that all language model 
    wrappers must implement, regardless of the underlying model provider.

    All LLM agent implementations should inherit from this class and
    implement the required methods for consistency across the codebase.
    """

    # noinspection PyMethodParameters
    @abstractmethod
    def response():
        """
        Abstract method for generating a single response from the LLM.

        Must be implemented by subclasses to handle sending a prompt
        to the LLM and processing its response.
        """
        pass

    # noinspection PyMethodParameters
    @abstractmethod
    def multi_responses():
        """
        Abstract method for generating multiple responses from the LLM.

        Must be implemented by subclasses to handle batch processing
        of multiple prompts, typically for DataFrame operations.
        """
        pass

class GPT_Agent(LLM_Agent):
    """
    Agent class for interacting with OpenAI's GPT models.

    This class provides a standardized interface for sending prompts to
    various GPT models through the OpenAI API, handling all the necessary
    parameters and formatting requirements.

    Supports different GPT model families including GPT-4 and GPT-5 variants
    with appropriate parameter adjustments for each family.
    """

    def __init__(self, OpenAIclient, SystemContent, model="gpt-4o-mini", max_tokens=16384, top_p=1, temperature=0, frequency_penalty=0, presence_penalty=0, logprobs=False, top_logprobs=None):
        """
        Initialize a GPT agent with the specified parameters.

        Args:
            OpenAIclient: An initialized OpenAI client object
            SystemContent: The system prompt to use for all interactions
            model: The GPT model to use (default: "gpt-4o-mini")
            max_tokens: Maximum tokens in the response (default: 16384)
            top_p: Nucleus sampling parameter (default: 1)
            temperature: Randomness parameter (default: 0 for deterministic outputs)
            frequency_penalty: Penalty for token frequency (default: 0)
            presence_penalty: Penalty for token presence (default: 0)
            logprobs: Whether to return log probabilities (default: False)
            top_logprobs: Number of top log probabilities to return (default: None)
        """
        self.OpenAIclient = OpenAIclient
        self.SystemContent = SystemContent
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

    # --- helper: build args that match the model family ---
    def _prepare_chat_create_args(self, messages, **kwargs):
        """
        Prepares API arguments based on the specific model family requirements.

        Args:
            messages: List of message dictionaries (system, user content)
            **kwargs: Additional parameters that can override defaults

        Returns:
            Dictionary of parameters appropriate for the selected model family

        Note:
            Automatically detects GPT-5 models and adjusts parameters accordingly
            For GPT-5: uses max_completion_tokens instead of max_tokens
            For non-GPT-5: includes temperature and uses max_tokens
        """
        model = kwargs.get("model", self.model)
        is_gpt5 = bool(re.search(r"\bgpt-5", model, flags=re.IGNORECASE))

        # Pull shared values (some may be None; we'll drop Nones)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        base = {
            "model": model,
            "messages": messages,
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs
        }

        # Family-specific params
        if is_gpt5:
            # For GPT-5 family: no temperature; use max_completion_tokens
            base["max_completion_tokens"] = max_tokens
        else:
            # For non-GPT-5: include temperature; use max_tokens
            base["temperature"] = kwargs.get("temperature", self.temperature)
            base["max_tokens"] = max_tokens

        # Drop any None entries to avoid passing invalid params
        return {k: v for k, v in base.items() if v is not None}

    # noinspection PyMethodOverriding
    def response(self, UserContent, InputWrapper, OutputWrapper, **kwargs):
        """
        Generates a single response from an Ollama-hosted model.

        Args:
            UserContent: The user's input text or a DataFrame row
            InputWrapper: Function to pre-process the input
            OutputWrapper: Function to post-process the output
            **kwargs: Additional parameters that can override defaults

        Returns:
            Processed response from the language model

        Note:
            Uses the ollama.chat API rather than OpenAI's API
            Formats messages in the structure expected by Ollama
            Returns just the content of the response message
        """
        defaultGPTparam = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        kwargs = {**defaultGPTparam, **kwargs}

        UserContent = InputWrapper(UserContent)
        messages = [
            {"role": "system", "content": self.SystemContent},
            {"role": "user", "content": UserContent}
        ]

        api_args = self._prepare_chat_create_args(messages, **kwargs)
        response = self.OpenAIclient.chat.completions.create(**api_args)

        output = OutputWrapper(response, **kwargs)
        return output

    # noinspection PyMethodOverriding
    def multi_responses(self, UserContentData, InputWrapper, OutputWrapper, **kwargs):
        """
        Processes multiple inputs through the LLM in sequence, managing token usage.

        Args:
            UserContentData: DataFrame containing multiple inputs to process
            InputWrapper: Function to pre-process each input
            OutputWrapper: Function to post-process each output
            **kwargs: Additional parameters including optional 'header' for limiting rows

        Returns:
            DataFrame containing all processed responses

        Note:
            Tracks token usage with a rolling 60-second window
            Skips posts that exceed token limits (>120,000 tokens)
            Pauses when hitting rate limits (10M tokens per minute)
            Reports progress for long-running batches
        """
        if kwargs.get('header') is not None:
            input_data = UserContentData.head(kwargs.get('header'))
        else:
            input_data = UserContentData
        too_big = []
        output_dict = {}
        token_roller = 0
        time_token_tracker = deque()
        counter = 0
        for index, row in input_data.iterrows():
            counter += 1
            if counter % 100 == 0:
                print(f"On Post: {counter}, roughly {round(counter / len(input_data) * 100, 0)}% done.")
            if row['word_count'] == 0:
                result_string = 'An error occurred: Error the error is there is no post self_text'
                response_df = OutputWrapper(result_string, **kwargs)
            else:
                tokens = int(row['word_count'] * 2)
                if tokens > 120000:
                    too_big.append(row['name'])
                    continue
                current_time = time.time()
                time_token_tracker.append((current_time, tokens))
                remove_old_tokens(time_token_tracker)
                token_roller = sum(t[1] for t in time_token_tracker)
                if token_roller >= 10000000:
                    print("Token limit reached, waiting for a minute...")
                    time.sleep(60)
                    remove_old_tokens(time_token_tracker)
                response_df = self.response(row, InputWrapper, OutputWrapper, **kwargs)
            output_dict[row['name']] = response_df
        output_df = pd.concat([df.assign(name=name) for name, df in output_dict.items()], ignore_index=True)
        return output_df

class Ollama_Agent(LLM_Agent):
    """
    Agent class for interacting with locally-hosted models through Ollama.

    This class provides an interface for open-source language models like
    Llama, Gemma, etc. running through the Ollama server. It follows the
    same interface pattern as GPT_Agent for consistency.

    Handles the specific API format and parameters required by the Ollama API
    while maintaining the same general workflow as other LLM agents.
    """
    def __init__(self, SystemContent, model=None, max_tokens=16384, top_p=1, temperature=0, frequency_penalty=0, presence_penalty=0):
        """
        Initialize an Ollama agent with the specified parameters.

        Args:
            SystemContent: The system prompt to use for all interactions
            model: The Ollama model to use (e.g., 'llama3', 'gemma:7b')
            max_tokens: Maximum tokens in the response (default: 16384)
            top_p: Nucleus sampling parameter (default: 1)
            temperature: Randomness parameter (default: 0 for deterministic outputs)
            frequency_penalty: Penalty for token frequency (default: 0)
            presence_penalty: Penalty for token presence (default: 0)
        """
        self.SystemContent = SystemContent
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
    # noinspection PyMethodOverriding
    def response(self, UserContent, InputWrapper, OutputWrapper, **kwargs):
        """
        Generates a single response from a GPT model.

        Args:
            UserContent: The user's input text or a DataFrame row
            InputWrapper: Function to pre-process the input
            OutputWrapper: Function to post-process the output
            **kwargs: Additional parameters that can override defaults

        Returns:
            Processed response from the language model

        Note:
            Uses the system content set during initialization
            Automatically handles different model families through _prepare_chat_create_args
            Can process both simple strings and DataFrame row objects
        """
        defaultGPTparam = {
            "model":self.model,
            "options": {
                "temperature":self.temperature,
                "top_p":self.top_p,
                "presence_penalty":self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "max_tokens": self.max_tokens,
                "reset": True
            }
        }
        kwargs = {**defaultGPTparam, **kwargs}
        UserContent = InputWrapper(UserContent)
        response = ollama.chat(model=kwargs.get('model'), messages=[
        {
        'role': 'system',
        'content': self.SystemContent,
        },
        {
        'role': 'user',
        'content': UserContent,
        },
        ],options=kwargs.get('options'))
        res = response['message']['content']
        output = OutputWrapper(res, **kwargs)
        return output

    # noinspection PyMethodOverriding
    def multi_responses(self, UserContentData, InputWrapper, OutputWrapper, **kwargs):
        """
        Processes multiple inputs through the Ollama model in sequence, managing token usage.

        Args:
            UserContentData: DataFrame containing multiple inputs to process
            InputWrapper: Function to pre-process each input
            OutputWrapper: Function to post-process each output
            **kwargs: Additional parameters including optional 'header' for limiting rows

        Returns:
            DataFrame containing all processed responses

        Note:
            Similar to GPT_Agent.multi_responses but adapted for Ollama API
            Tracks token usage with a rolling 60-second window
            Skips posts that exceed token limits (>120,000 tokens)
            Pauses when hitting rate limits (10M tokens per minute)
        """
        if kwargs.get('header') is not None:
            input_data = UserContentData.head(kwargs.get('header'))
        else:
            input_data = UserContentData
        too_big = []
        output_dict = {}
        token_roller = 0
        time_token_tracker = deque()
        counter = 0
        for index, row in input_data.iterrows():
            counter += 1
            if counter % 100 == 0:
                print(f"On Post: {counter}, roughly {round(counter / len(input_data) * 100, 0)}% done.")
            if row['word_count'] == 0:
                result_string = 'An error occurred: Error the error is there is no post self_text'
                response_df = OutputWrapper(result_string, **kwargs)
            else:
                tokens = int(row['word_count']*2)# approximate tokens with 2 since we have a huge instruction prompt and are missing title
                if tokens > 120000:
                    too_big.append(row['name'])
                    continue # Skip posts that are too big
                current_time = time.time()
                time_token_tracker.append((current_time, tokens))
                remove_old_tokens(time_token_tracker)
                token_roller = sum(t[1] for t in time_token_tracker)
                if token_roller >= 10000000:
                    print("Token limit reached, waiting for a minute...")
                    time.sleep(60)
                    remove_old_tokens(time_token_tracker)
                response_df = self.response(row, InputWrapper, OutputWrapper, **kwargs)
            output_dict[row['name']] = response_df
        output_df = pd.concat([df.assign(name=name) for name, df in output_dict.items()], ignore_index=True)
        return output_df

if __name__ == "__main__":
    OpenAIclient_ = OpenAI(api_key=GPT_API_KEY)
    print('Hello World')