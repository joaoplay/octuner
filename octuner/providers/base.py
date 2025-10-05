import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Standard response format from all LLM providers.
    """
    text: str
    provider: Optional[str] = None
    model: Optional[str] = None
    cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for implementing custom LLM providers in Octuner.
    
    This class serves as the foundation for creating custom LLM provider implementations,
    enabling to integrate own self-hosted models, proprietary APIs, or any other LLM
    service.
    
    Key Features:
    - **Configuration-driven**: Integrates with YAML-based configuration system
    - **Parameter optimization**: Supports automatic parameter tuning through the config loader
    - **Type conversion**: Automatic parameter type conversion based on configuration
    - **Cost tracking**: Built-in cost calculation and token usage tracking
    - **Standardized responses**: Returns consistent LLMResponse objects across all providers

    To create a custom provider, you must:
    
    1. **Inherit from BaseLLMProvider** and set the `provider_name` attribute
    2. **Implement abstract methods**:
       - `call()`: Main interface for making LLM requests
       - `_make_request()`: Low-level API communication
       - `_parse_response()`: Convert raw API response to LLMResponse
       - `_calculate_cost()`: Calculate cost based on token usage
    
    3. **Create a configuration file** (YAML) defining:
       - Available models and their parameters
       - Parameter types, ranges, and defaults
       - Pricing information for cost calculation
       - Provider-specific settings
    
    Example Usage:
    ```python
    from octuner.providers.base import BaseLLMProvider, LLMResponse
    from octuner.config.loader import ConfigLoader
    
    class CustomProvider(BaseLLMProvider):
        def __init__(self, config_loader, **kwargs):
            super().__init__(config_loader, **kwargs)
            self.provider_name = "custom"
            # Initialize your API client here
        
        def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
            # Implementation details...
            pass
        
        # Implement other abstract methods...
    
    # Usage with configuration
    config_loader = ConfigLoader("my_custom_config.yaml")
    provider = CustomProvider(config_loader, api_key="your-key")
    response = provider.call("Hello, world!")
    ```
    
    Configuration File Structure:
    ```yaml
    providers:
      custom:
        default_model: "my-model-v1"
        available_models: ["my-model-v1", "my-model-v2"]
        pricing_usd_per_1m_tokens:
          my-model-v1: [0.5, 1.0]  # [input_cost, output_cost]
        model_capabilities:
          my-model-v1:
            supported_parameters: ["temperature", "max_tokens"]
            parameters:
              temperature:
                type: float
                range: [0.0, 2.0]
                default: 0.7
              max_tokens:
                type: int
                range: [1, 4000]
                default: 1000
    ```
    """

    def __init__(self, config_loader, **kwargs):
        """
        This constructor sets up the provider with access to the configuration system
        and stores any provider-specific parameters. The config_loader is mandatory
        as it provides access to model capabilities, parameter definitions, and pricing.
        
        Args:
            config_loader (ConfigLoader): Configuration loader instance that provides
                access to YAML configuration files. This is mandatory.
                
            **kwargs: Provider-specific configuration parameters. These can override
                default values from the configuration file.
        """
        if config_loader is None:
            raise ValueError("config_loader is mandatory for all providers")
        
        self.config = kwargs
        self.config_loader = config_loader
        self.provider_name = getattr(self, 'provider_name', None)  # Set by subclasses

    def _get_parameter(self, param_name: str, kwargs: Dict[str, Any], model: str) -> Any:
        """
        This method is used in the `call()` method to resolve parameter values before
        making API requests. It enables the optimization framework to override default
        parameters during tuning.
        
        Args:
            param_name (str): Name of the parameter to retrieve
            kwargs (Dict[str, Any]): Keyword arguments that may contain the parameter
            model (str): Model identifier for parameter-specific defaults
        """
        config_value = self.config_loader.get_parameter_default(self.provider_name, model, param_name)
        if param_name in kwargs:
            logger.debug(f"Overriding {param_name} from config ({config_value}) with kwargs ({kwargs[param_name]})")
            return kwargs[param_name]
        return config_value

    def _convert_parameter_type(self, param_name: str, value: Any, model: str) -> Any:
        """
        Performs automatic type conversion for parameters based on the
        expected type defined in the configuration file. It ensures that parameters
        are properly typed before being passed to the underlying API.

        Supported Types:
        - **int**: Converts to integer, empty string becomes None
        - **float**: Converts to float, empty string becomes None
        - **str**: Converts to string, empty string becomes None
        - **bool**: Converts various representations to boolean:
          - String: "true", "1", "yes", "on" → True; others → False
          - Boolean: Passes through unchanged
          - Other: Uses bool() conversion
        - **choice**: Returns value as-is, empty string becomes None
        - **list**: Converts to list:
          - List: Passes through unchanged
          - String: Attempts JSON parsing, falls back to comma-separated split
          - Other iterable: Converts to list
        
        Args:
            param_name (str): Name of the parameter being converted
            value (Any): Raw parameter value to convert
            model (str): Model identifier for type lookup
        
        Returns:
            Any: Converted value with proper type, or None for empty values
        """
        if value is None:
            return None
        expected_type = self.config_loader.get_parameter_type(self.provider_name, model, param_name)
        try:
            if expected_type == "int":
                return int(value) if value != "" else None
            elif expected_type == "float":
                return float(value) if value != "" else None
            elif expected_type == "str":
                return str(value) if value != "" else None
            elif expected_type == "bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif expected_type == "choice":
                return value if value != "" else None
            elif expected_type == "list":
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return [item.strip() for item in value.split(",") if item.strip()]
                return list(value) if value else []
            else:
                raise ValueError(f"Unknown parameter type '{expected_type}' for {param_name} in {self.provider_name}:{model}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert {param_name}={value} to {expected_type} for {self.provider_name}:{model}: {e}")

    @abstractmethod
    def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        This is the main interface method that users will call to interact with
        the LLM provider. It handles parameter resolution, type conversion,
        API communication, and response parsing.
        
        Args:
            prompt (str): The user prompt/query to send to the LLM
            system_prompt (Optional[str]): Optional system prompt that sets the
                context or behavior for the LLM. If None, no system prompt is used.
            **kwargs: Additional parameters that can override configuration defaults.
        
        Returns:
            LLMResponse: Standardized response object containing:
        
        Example Implementation:
            ```python
            def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
                import time
                start_time = time.time()
                
                # Get model and parameters
                model = self._get_parameter("model", kwargs, "default-model")
                temperature = self._get_parameter("temperature", kwargs, model)
                max_tokens = self._get_parameter("max_tokens", kwargs, model)
                
                # Convert types
                temperature = self._convert_parameter_type("temperature", temperature, model)
                max_tokens = self._convert_parameter_type("max_tokens", max_tokens, model)
                
                # Prepare API request
                api_params = {
                    "model": model,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Make API call
                response = self._make_request(**api_params)
                
                # Parse and return response
                result = self._parse_response(response)
                result.latency_ms = (time.time() - start_time) * 1000
                return result
            ```
        """
        pass

    @abstractmethod
    def _make_request(self, **kwargs) -> Any:
        """
        Make the actual API request to the service.
        
        Args:
            **kwargs: Request parameters that have been processed and type-converted.
        """
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> LLMResponse:
        """
        Converts the provider-specific API response into standardized LLMResponse format.
        It extracts text content, metadata, and token usage information from the raw response.

        1. **Text Extraction**: Extract the generated text from the response
        2. **Token Counting**: Extract input and output token counts if available
        3. **Metadata**: Include relevant response metadata
        4. **Error Handling**: Handle cases where expected fields are missing
        5. **Provider Info**: Set provider name and model information
        
        Returns:
            LLMResponse: Standardized response object with:
        """
        pass

    @abstractmethod
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """
        Calculate the estimated cost of an API call based on the number of tokens consumed
        and the pricing information from the configuration. The cost calculation is
        essential for optimization algorithms that consider cost as a factor in their
        decision-making.
        
        Implementation Guidelines:
        1. **Pricing Lookup**: Use `get_cost_per_token()` to get pricing rates
        2. **Token Calculation**: Apply pricing to input and output tokens separately
        3. **Unit Conversion**: Convert from per-1M-tokens to actual cost
        4. **Error Handling**: Handle cases where pricing is not available
        
        Args:
            input_tokens (int): Number of input tokens consumed by the request
            output_tokens (int): Number of output tokens generated by the model
            model (str): Model identifier for pricing lookup
        
        Returns:
            float: Total cost for the API call
        
        Example Implementation:
            ```python
            def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
                # Get pricing rates from configuration
                input_cost_per_1m, output_cost_per_1m = self.get_cost_per_token(model)
                
                # Calculate costs
                input_cost = (input_tokens * input_cost_per_1m) / 1_000_000
                output_cost = (output_tokens * output_cost_per_1m) / 1_000_000
                
                return input_cost + output_cost
            ```
        """
        pass

    def get_cost_per_token(self, model: str) -> Tuple[float, float]:
        """
        Get the cost per input and output token for a model.
        
        Args:
            model: Model identifier
            
        Returns:
            Tuple of (input_cost_per_1M_tokens, output_cost_per_1M_tokens)
        """
        return self.config_loader.get_pricing(self.provider_name, model)

    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model identifiers
        """
        return self.config_loader.get_available_models(self.provider_name)
