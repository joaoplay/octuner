import logging
from typing import Dict, List, Tuple, Type

from .base import BaseLLMProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider

logger = logging.getLogger(__name__)

# Provider registry
PROVIDERS = {
    'openai': OpenAIProvider,
    'gemini': GeminiProvider,
}


def register_provider(name: str, provider_class: Type[BaseLLMProvider]) -> None:
    """
    Register a custom LLM provider.
    
    This allows you to add custom providers for self-hosted LLMs or other services.
    
    Args:
        name: Provider name (e.g., 'ollama', 'vllm', 'custom')
        provider_class: Provider class that inherits from BaseLLMProvider
        
    Raises:
        ValueError: If provider_class doesn't inherit from BaseLLMProvider
    """
    if not issubclass(provider_class, BaseLLMProvider):
        raise ValueError(
            f"Provider class must inherit from BaseLLMProvider, "
            f"got {provider_class.__name__}"
        )
    
    if name in PROVIDERS:
        logger.warning(f"Overwriting existing provider: {name}")
    
    PROVIDERS[name] = provider_class
    logger.info(f"Registered provider: {name} ({provider_class.__name__})")


def unregister_provider(name: str) -> None:
    """
    Unregister a provider.
    
    Args:
        name: Provider name to remove
        
    Raises:
        KeyError: If provider is not registered
    """
    if name not in PROVIDERS:
        raise KeyError(f"Provider not registered: {name}")
    
    del PROVIDERS[name]
    logger.info(f"Unregistered provider: {name}")


def list_providers() -> List[str]:
    """
    Get list of all registered provider names.
    
    Returns:
        List of provider names
    """
    return list(PROVIDERS.keys())


def get_provider(provider_name: str, config_loader, **kwargs) -> BaseLLMProvider:
    """
    Get a provider instance by name.
    
    Args:
        provider_name: Name of the provider ('openai', 'gemini', or custom)
        config_loader: ConfigLoader for configuration-driven behavior (mandatory)
        **kwargs: Provider-specific configuration
        
    Returns:
        Provider instance
        
    Raises:
        KeyError: If provider is not supported
    """
    if provider_name not in PROVIDERS:
        raise KeyError(
            f"Unsupported provider: {provider_name}. "
            f"Available: {list_providers()}"
        )

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
