import logging
from typing import Any, Dict, Optional, Tuple
from .mixin import TunableMixin
from .registry import register_tunable_class
from ..providers import BaseLLMProvider, LLMResponse, get_provider
from ..config.loader import ConfigLoader

logger = logging.getLogger(__name__)


class MultiProviderTunableLLM(TunableMixin):
    """
    A tunable LLM wrapper that optimizes provider, model, and parameter selection.
    
    This class allows the optimization system to discover the best combination of:
    - LLM provider (OpenAI, Gemini, etc.)
    - Model within that provider
    - Model-specific parameters (temperature, max_tokens, etc.)
    
    Configuration is defined explicitly via YAML files.
    
    Example:
        llm = MultiProviderTunableLLM(config_file="my_llm_config.yaml")
        response = llm.call("What is the capital of France?")
        print(response.text)
    """

    def __init__(self, config_file: str, default_provider: str = "openai", default_model: Optional[str] = None,
                 provider_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the tunable LLM with explicit configuration.
        
        Args:
            config_file: Path to YAML configuration file (required)
            default_provider: Default provider to use ('openai', 'gemini') 
            default_model: Default model to use (if None, uses provider's default)
            provider_configs: Configuration for each provider (API keys, etc.)
        """
        # Initialize TunableMixin
        super().__init__()
        
        # Initialize configuration
        self.config_loader = ConfigLoader(config_file)
        self.config_loader.validate_config()

        # Use config file settings
        if default_model is None:
            default_model = self.config_loader.get_default_model(default_provider)

        logger.info(f"Using config file: {config_file}")

        self.default_provider = default_provider
        self.default_model = default_model
        self.provider_configs = provider_configs or {}

        # Current settings (will be modified by optimiser)
        self.provider_name = default_provider
        self.model = default_model

        # Auto-discover and initialise all parameters from config
        # Values will be set by the optimiser during tuning
        self._initialize_parameters_from_config(default_provider, default_model)
        
        # Initialize provider_model attribute for tuning
        self.provider_model = f"{default_provider}:{default_model}"
        
        # Initialize websearch parameters
        self.use_websearch = False
        self.search_context_size = 5

        # Cache for provider instances
        self._provider_cache: Dict[str, BaseLLMProvider] = {}

        # Generate tunable parameter space
        self._setup_tunables()

    def _setup_tunables(self):
        """
        Set up the tunable parameter space from explicit config file.
        """
        self._setup_tunables_from_config()

    def _initialize_parameters_from_config(self, default_provider: str, default_model: str):
        """
        Dynamically initialize all parameters from config file.
        """
        # Get all parameters supported by the default model
        supported_params = self.config_loader.get_supported_parameters(default_provider, default_model)

        # Set each parameter as an attribute with its default value
        for param in supported_params:
            default_value = self.config_loader.get_parameter_default(default_provider, default_model, param)
            setattr(self, param, default_value)
            logger.debug(f"Initialized parameter '{param}' = {default_value}")

    def _setup_tunables_from_config(self):
        """
        Set up tunable parameters using automatic discovery from config file.
        """
        # Get available providers and models from config
        providers = self.config_loader.get_providers()
        combinations = []

        for provider in providers:
            models = self.config_loader.get_available_models(provider)
            for model in models:
                combinations.append((provider, model))

        provider_model_choices = [f"{provider}:{model}" for provider, model in combinations]

        # Define tunable parameters - start with provider/model selection
        params = {
            "provider_model": ("choice", provider_model_choices),
        }

        # Auto-discover all parameters from any model
        all_parameters = set()
        for provider, model in combinations:
            model_params = self.config_loader.get_supported_parameters(provider, model)
            all_parameters.update(model_params)

        # Add websearch parameters for providers that support it (OpenAI and Gemini)
        for provider, model in combinations:
            if provider in ["openai", "gemini"]:  # Both OpenAI and Gemini support websearch
                all_parameters.add("use_websearch")
                all_parameters.add("search_context_size")

        logger.info(f"Auto-discovered parameters: {sorted(all_parameters)}")

        # Add each discovered parameter to tunable space
        for param in all_parameters:
            for provider, model in combinations:
                if self.config_loader.model_supports_parameter(provider, model, param):
                    param_type = self.config_loader.get_parameter_type(provider, model, param)
                    param_range = self.config_loader.get_parameter_range(provider, model, param)

                    if param_type in ['float', 'int']:
                        # For numeric types, use min/max range
                        params[param] = (param_type, param_range[0], param_range[1])
                    else:
                        # For choice, list, bool types, use the range as choices
                        params[param] = (param_type, param_range)
                    break

        # Register this class as tunable
        if not params:
            raise ValueError("No tunable parameters found in configuration.")

        register_tunable_class(self.__class__, params, call_method="call")
        
        # Also populate _tunable_params for compatibility with TunableMixin methods
        for param_name, param_def in params.items():
            if len(param_def) >= 3:
                # Numeric types with (type, min, max)
                self._tunable_params[param_name] = {
                    "type": param_def[0],
                    "range": (param_def[1], param_def[2]),
                    "default": None  # We don't have a default from the config
                }
            else:
                # Choice types with (type, choices)
                self._tunable_params[param_name] = {
                    "type": param_def[0],
                    "range": param_def[1],
                    "default": param_def[1][0] if param_def[1] else None
                }

    def _get_provider(self, provider_name: str) -> BaseLLMProvider:
        """
        Get a provider instance, using cache for efficiency.
        """
        if provider_name not in self._provider_cache:
            config = self.provider_configs.get(provider_name, {})
            self._provider_cache[provider_name] = get_provider(provider_name, config_loader=self.config_loader,
                                                               **config)
        return self._provider_cache[provider_name]

    def _parse_provider_model(self, provider_model: str) -> Tuple[str, str]:
        """
        Parse provider_model string into provider and model names.
        """
        if ":" in provider_model:
            provider, model = provider_model.split(":", 1)
            return provider, model
        else:
            # Fallback for backward compatibility
            return self.default_provider, provider_model

    def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Make an LLM call using the current provider and parameters.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters that override instance settings
            
        Returns:
            LLMResponse with the result
        """
        # Parse current provider and model
        provider_model = getattr(self, 'provider_model', f"{self.provider_name}:{self.model}")
        provider_name, model = self._parse_provider_model(provider_model)

        # Get provider instance
        provider = self._get_provider(provider_name)

        # Auto-collect all parameter attributes
        params = {'model': model}

        # Get all supported parameters for this model and collect their values
        supported_params = self.config_loader.get_supported_parameters(provider_name, model)
        for param in supported_params:
            if hasattr(self, param):
                params[param] = getattr(self, param)
                logger.debug(f"Using parameter '{param}' = {params[param]}")

        # Handle websearch parameters for providers that support it
        if provider_name in ["openai", "gemini"]:
            if hasattr(self, 'use_websearch'):
                params['use_websearch'] = getattr(self, 'use_websearch')
                logger.debug(f"Using websearch parameter: {params['use_websearch']}")
            
            if hasattr(self, 'search_context_size'):
                params['search_context_size'] = getattr(self, 'search_context_size')
                logger.debug(f"Using search_context_size parameter: {params['search_context_size']}")

        # Override with any provided kwargs
        params.update(kwargs)

        # Make the call
        try:
            response = provider.call(prompt, system_prompt, **params)
            logger.debug(f"LLM call successful: {provider_name}:{model}")
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {provider_name}:{model} - {e}")
            # Re-raise the exception so the optimizer can handle failures
            raise

    def llm_eq_cost(self, *, input_tokens=None, output_tokens=None, metadata=None):
        """
        Calculate the cost of an LLM call based on current provider and model.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Additional metadata from the LLM call
            
        Returns:
            Cost in USD, or None if tokens are not available
        """
        if input_tokens is None or output_tokens is None:
            return None

        # Parse current provider and model
        provider_model = getattr(self, 'provider_model', f"{self.provider_name}:{self.model}")
        provider_name, model = self._parse_provider_model(provider_model)

        try:
            # Get provider instance and cost information
            provider = self._get_provider(provider_name)
            input_cost_per_1m, output_cost_per_1m = provider.get_cost_per_token(model)

            # Calculate cost
            input_cost = (input_tokens / 1e6) * input_cost_per_1m
            output_cost = (output_tokens / 1e6) * output_cost_per_1m

            return input_cost + output_cost
        except Exception as e:
            logger.warning(f"Could not calculate cost for {provider_name}:{model} - {e}")
            return None

    def get_current_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider and model.
        
        Returns:
            Dictionary with provider info
        """
        provider_model = getattr(self, 'provider_model', f"{self.provider_name}:{self.model}")
        provider_name, model = self._parse_provider_model(provider_model)

        try:
            provider = self._get_provider(provider_name)
            input_cost, output_cost = provider.get_cost_per_token(model)

            return {
                'provider': provider_name,
                'model': model,
                'cost_per_1m_input_tokens': input_cost,
                'cost_per_1m_output_tokens': output_cost,
            }
        except Exception as e:
            return {
                'provider': provider_name,
                'model': model,
                'error': str(e)
            }

    def set_provider_configs(self, configs: Dict[str, Dict[str, Any]]):
        """
        Update provider configurations (API keys, base URLs, etc.).
        
        Args:
            configs: Dictionary mapping provider names to configuration dicts
        """
        self.provider_configs.update(configs)
        # Clear cache to force recreation with new configs
        self._provider_cache.clear()
