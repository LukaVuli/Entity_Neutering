import os

# Try to get API key from environment variable first, then fall back to hardcoded value
# Users can set this via: export OPENAI_API_KEY='your-key-here'
# Or by directly setting GPT_API_KEY below
GPT_API_KEY = os.getenv('OPENAI_API_KEY', '')

# If you prefer to hardcode your API key, uncomment and fill in the line below:
# GPT_API_KEY = 'your-api-key-here'