import logging
from typing import Dict, List, Tuple

from .base import BaseLLMProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider

logger = logging.getLogger(__name__)

# Provider registry
PROVIDERS = {
    'openai': OpenAIProvider,
    'gemini': GeminiProvider,
}


def get_provider(provider_name: str, config_loader, **kwargs) -> BaseLLMProvider:
    """
    Get a provider instance by name.
    
    Args:
        provider_name: Name of the provider ('openai', 'gemini')
        config_loader: ConfigLoader for configuration-driven behavior (mandatory)
        **kwargs: Provider-specific configuration
        
    Returns:
        Provider instance
        
    Raises:
        KeyError: If provider is not supported
    """
    if provider_name not in PROVIDERS:
        raise KeyError(f"Unsupported provider: {provider_name}. Available: {list(PROVIDERS.keys())}")

    return PROVIDERS[provider_name](config_loader=config_loader, **kwargs)


def get_all_models() -> Dict[str, List[str]]:
    """
    Get all available models grouped by provider.
    
    Returns:
        Dictionary mapping provider names to lists of available models
    """
    all_models = {}
    for provider_name, provider_class in PROVIDERS.items():
        try:
            # Create a dummy config loader for this operation
            from ..config.loader import ConfigLoader
            # This is a bit hacky - we need a config file to get models
            # In practice, this should be called with a proper config loader
            # For now, return some default models
            if provider_name == 'openai':
                all_models[provider_name] = ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']
            elif provider_name == 'gemini':
                all_models[provider_name] = ['gemini-1.5-flash', 'gemini-1.5-pro']
            else:
                all_models[provider_name] = []
        except Exception as e:
            logger.warning(f"Could not get models for {provider_name}: {e}")
            all_models[provider_name] = []

    return all_models


def get_all_provider_model_combinations() -> List[Tuple[str, str]]:
    """
    Get all possible (provider, model) combinations.
    
    Returns:
        List of (provider_name, model_name) tuples
    """
    combinations = []
    all_models = get_all_models()

    for provider_name, models in all_models.items():
        for model in models:
            combinations.append((provider_name, model))

    return combinations
