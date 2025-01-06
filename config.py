from typing import Dict

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant that provides accurate and concise responses."

# Group models by provider for UI organization
MODEL_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "o1-mini"]
    },
    "anthropic": {
        "name": "Anthropic",
        "models": ["claude-3-opus-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
    },
    "nvidia": {
        "name": "NVIDIA",
        "models": ["nvidia/llama-3.1-nemotron-70b-instruct"]
    }
}

# Initialize all prompts with the default
DEFAULT_SYSTEM_PROMPTS = {model_id: DEFAULT_SYSTEM_PROMPT for provider in MODEL_PROVIDERS.values() for model_id in provider["models"]}

MODEL_CONFIGS = {
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Versatile, high-intelligence flagship model",
        "provider": "openai",
        "model": "gpt-4o",
    },
    "o1-mini": {
        "name": "O1-Mini",
        "description": "Fast and affordable reasoning model",
        "provider": "openai",
        "model": "o1-mini",
    },
    "claude-3-opus-latest": {
        "name": "Claude 3 Opus",
        "description": "Powerful for highly complex tasks",
        "provider": "anthropic",
        "model": "claude-3-opus-latest",
    },
    "claude-3-5-sonnet-latest": {
        "name": "Claude 3.5 Sonnet",
        "description": "Most intelligent",
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-latest",
    },
    "claude-3-5-haiku-latest": {
        "name": "Claude 3.5 Haiku",
        "description": "Fastest",
        "provider": "anthropic",
        "model": "claude-3-5-haiku-latest",
    },
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "name": "Nemotron 70B",
        "description": "API limited to 4000 queries only",
        "provider": "nvidia",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "base_url": "https://integrate.api.nvidia.com/v1"
    }
} 