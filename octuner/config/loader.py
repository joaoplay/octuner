"""
Configuration file loader for Octuner.

This module provides utilities to load YAML configuration files explicitly
provided by engineers using the library.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and validates YAML configuration files.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize config loader with a specific file.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = Path(config_file)
        self._config: Optional[Dict[str, Any]] = None
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        # Although not the best to have IO inside constructor, I'm doing it here to avoid someone to forget loading
        # the config after creating the object.
        self._load_config()
    
    def _load_config(self):
        """
        Load the YAML configuration file.
        """
        try:
            with open(self.config_file, 'r') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {self.config_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config file {self.config_file}: {e}")
    
    def get_providers(self) -> List[str]:
        """
        Get list of available providers.
        """
        return list(self._config.get('providers', {}).keys())
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        """
        providers = self._config.get('providers', {})
        if provider_name not in providers:
            raise ValueError(f"Provider '{provider_name}' not found in config. Available: {list(providers.keys())}")
        return providers[provider_name]
    
    def get_default_model(self, provider_name: str) -> str:
        """
        Get default model for a provider.
        """
        provider_config = self.get_provider_config(provider_name)
        if 'default_model' not in provider_config:
            raise ValueError(f"No default_model specified for provider '{provider_name}'")
        return provider_config['default_model']
    
    def get_available_models(self, provider_name: str) -> List[str]:
        """
        Get available models for a provider.
        """
        provider_config = self.get_provider_config(provider_name)
        return provider_config.get('available_models', [])
    
    def get_pricing(self, provider_name: str, model: str) -> Tuple[float, float]:
        """
        Get pricing for a model (input_cost, output_cost per 1M tokens).
        """
        provider_config = self.get_provider_config(provider_name)
        pricing = provider_config.get('pricing_usd_per_1m_tokens', {})
        
        if model not in pricing:
            logger.warning(f"No pricing info for {provider_name}:{model}, using fallback")
            return 0.15, 0.60  # Fallback pricing
        
        return tuple(pricing[model])
    
    def get_model_capabilities(self, provider_name: str, model: str) -> Dict[str, Any]:
        """
        Get capabilities for a specific model.
        """
        provider_config = self.get_provider_config(provider_name)
        model_capabilities = provider_config.get('model_capabilities', {})
        
        if model not in model_capabilities:
            raise ValueError(
                f"Model '{model}' not found in capabilities for provider '{provider_name}'. "
                f"Available models: {list(model_capabilities.keys())}"
            )
        
        return model_capabilities[model]
    
    def get_supported_parameters(self, provider_name: str, model: str) -> List[str]:
        """
        Get list of parameters that can be optimized for a model.
        """
        capabilities = self.get_model_capabilities(provider_name, model)
        return capabilities.get('supported_parameters', [])
    
    def model_supports_parameter(self, provider_name: str, model: str, parameter: str) -> bool:
        """
        Check if a model supports a specific parameter.
        """
        supported_params = self.get_supported_parameters(provider_name, model)
        return parameter in supported_params
    
    def get_parameter_range(self, provider_name: str, model: str, parameter: str) -> tuple[Any]:
        """
        Get optimization range for a parameter.
        """
        # Handle special case for 'use_websearch' parameter - boolean choices
        if parameter == 'use_websearch':
            return (True, False)
        
        # Handle special case for 'search_context_size' parameter - integer range
        if parameter == 'search_context_size':
            return (1, 20)  # Reasonable range for search context size
        
        capabilities = self.get_model_capabilities(provider_name, model)
        parameters = capabilities.get('parameters', {})
        
        if parameter not in parameters:
            raise ValueError(f"No parameter definition found for '{parameter}' in {provider_name}:{model}")
        
        param_config = parameters[parameter]
        
        # Handle choice type parameters
        if param_config.get('type') == 'choice':
            # For choice types, use the 'range' field as choices
            if 'range' not in param_config:
                raise ValueError(f"No range defined for choice parameter '{parameter}' in {provider_name}:{model}")
            return tuple(param_config['range'])
        
        # Handle range type parameters
        if 'range' not in param_config:
            raise ValueError(f"No range defined for parameter '{parameter}' in {provider_name}:{model}")
        
        return tuple(param_config['range'])
    
    def get_parameter_default(self, provider_name: str, model: str, parameter: str) -> Any:
        """
        Get default value for a parameter.
        
        Args:
            provider_name: Name of the provider
            model: Name of the model
            parameter: Name of the parameter
            
        Returns:
            Default parameter value from configuration
            
        Raises:
            ValueError: If parameter default is not defined in configuration
        """
        # Handle special case for 'model' parameter - return the current model
        if parameter == 'model':
            return model
        
        capabilities = self.get_model_capabilities(provider_name, model)
        parameters = capabilities.get('parameters', {})
        
        if parameter not in parameters:
            raise ValueError(f"No parameter definition found for '{parameter}' in {provider_name}:{model}")
        
        param_config = parameters[parameter]
        if 'default' not in param_config:
            raise ValueError(f"No default value defined for parameter '{parameter}' in {provider_name}:{model}")
        
        return param_config['default']
    
    def get_parameter_type(self, provider_name: str, model: str, parameter: str) -> str:
        """
        Get the expected type for a parameter from YAML configuration.
        
        Args:
            provider_name: Name of the provider
            model: Name of the model
            parameter: Name of the parameter
            
        Returns:
            Parameter type ('int', 'float', 'str', 'bool', 'choice')
            
        Raises:
            ValueError: If parameter type is not defined in configuration
        """
        # Handle special case for 'model' parameter - it's always a string
        if parameter == 'model':
            return 'str'
        
        capabilities = self.get_model_capabilities(provider_name, model)
        parameters = capabilities.get('parameters', {})
        
        if parameter in parameters:
            param_config = parameters[parameter]
            if 'type' in param_config:
                return param_config['type']
        
        # If no type is defined in YAML, raise an error to force explicit configuration
        raise ValueError(
            f"Parameter type for '{parameter}' not defined in configuration for {provider_name}:{model}. "
            f"Please add 'parameters' section to your YAML config with the type for this parameter."
        )
    
    def get_forced_parameter(self, provider_name: str, model: str, parameter: str) -> Optional[Any]:
        """
        Get forced value for a parameter (if any).
        """
        capabilities = self.get_model_capabilities(provider_name, model)
        forced = capabilities.get('forced_parameters', {})
        return forced.get(parameter)
    
    def validate_config(self) -> bool:
        """
        Validate the configuration structure.
        """
        if 'providers' not in self._config:
            raise ValueError("Configuration must have 'providers' section")
        
        for provider_name, provider_config in self._config['providers'].items():
            # Check required fields
            if 'default_model' not in provider_config:
                raise ValueError(f"Provider '{provider_name}' must have 'default_model'")
            
            # Validate model capabilities
            capabilities = provider_config.get('model_capabilities', {})
            for model, model_config in capabilities.items():
                if 'supported_parameters' not in model_config:
                    raise ValueError(f"Model '{model}' must have 'supported_parameters'")
                
                supported = model_config['supported_parameters']
                parameters = model_config.get('parameters', {})
                
                # Check that parameter definitions exist for supported parameters
                for param in supported:
                    if param not in parameters and param not in ['system_role', 'use_websearch', 'search_context_size']:  # these don't need definition
                        logger.warning(f"No parameter definition found for supported parameter '{param}' in {provider_name}:{model}")
                    elif param in parameters:
                        param_config = parameters[param]
                        # Check required fields
                        if 'type' not in param_config:
                            raise ValueError(f"Parameter '{param}' in {provider_name}:{model} must have 'type' defined")
                        if 'range' not in param_config:
                            raise ValueError(f"Parameter '{param}' in {provider_name}:{model} must have 'range' defined")
                        if 'default' not in param_config:
                            logger.warning(f"Parameter '{param}' in {provider_name}:{model} should have 'default' defined")
        
        logger.info("Configuration validation passed")
        return True
