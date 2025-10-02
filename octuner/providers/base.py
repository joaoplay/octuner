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
    enabling users to integrate their own self-hosted models, proprietary APIs, or any
    other LLM service into Octuner's optimization framework.
    
    Key Features:
    - **Configuration-driven**: Integrates with Octuner's YAML-based configuration system
    - **Parameter optimization**: Supports automatic parameter tuning through the config loader
    - **Type conversion**: Automatic parameter type conversion based on configuration
    - **Cost tracking**: Built-in cost calculation and token usage tracking
    - **Standardized responses**: Returns consistent LLMResponse objects across all providers
    
    Implementation Requirements:
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
    
    Notes:
    - The `config_loader` parameter is mandatory and provides access to configuration
    - Parameter values are automatically type-converted based on configuration
    - Cost calculation is handled automatically if pricing is configured
    - All providers return standardized `LLMResponse` objects
    - The framework handles parameter optimization and tuning automatically
    """

    def __init__(self, config_loader, **kwargs):
        """
        Initialize the provider with configuration and optional parameters.
        
        This constructor sets up the provider with access to the configuration system
        and stores any provider-specific parameters. The config_loader is mandatory
        as it provides access to model capabilities, parameter definitions, and pricing.
        
        Args:
            config_loader (ConfigLoader): Configuration loader instance that provides
                access to YAML configuration files. This is mandatory and cannot be None.
                The config loader enables:
                - Parameter type conversion and validation
                - Model capability discovery
                - Pricing information retrieval
                - Default parameter value resolution
                
            **kwargs: Provider-specific configuration parameters. These can override
                default values from the configuration file. Common parameters include:
                - `api_key` (str): API key for the provider service
                - `base_url` (str): Custom base URL for API endpoints
                - `timeout` (int): Request timeout in seconds
                - `model` (str): Default model to use
                - Any other provider-specific settings
        
        Raises:
            ValueError: If config_loader is None
        
        Example:
            ```python
            from octuner.config.loader import ConfigLoader
            
            # Basic initialization
            config_loader = ConfigLoader("my_config.yaml")
            provider = MyProvider(config_loader)
            
            # With provider-specific parameters
            provider = MyProvider(
                config_loader,
                api_key="your-api-key",
                base_url="https://api.example.com",
                timeout=60
            )
            ```
        
        Note:
            Subclasses should call `super().__init__(config_loader, **kwargs)` first,
            then set `self.provider_name` to match the provider name in the config file.
        """
        if config_loader is None:
            raise ValueError("config_loader is mandatory for all providers")
        
        self.config = kwargs
        self.config_loader = config_loader
        self.provider_name = getattr(self, 'provider_name', None)  # Set by subclasses

    def _get_parameter(self, param_name: str, kwargs: Dict[str, Any], model: str) -> Any:
        """
        Get parameter value with config loader fallback and kwargs override.
        
        This helper method implements the parameter resolution strategy:
        1. First check if the parameter is provided in kwargs (highest priority)
        2. If not found, get the default value from the configuration loader
        3. Return None if the parameter is not defined anywhere
        
        This method is typically used in the `call()` method to resolve parameter
        values before making API requests. It enables the optimization framework
        to override default parameters during tuning.
        
        Args:
            param_name (str): Name of the parameter to retrieve
            kwargs (Dict[str, Any]): Keyword arguments that may contain the parameter
            model (str): Model identifier for parameter-specific defaults
        
        Returns:
            Any: Parameter value from kwargs if present, otherwise from config,
                 or None if not defined
        
        Example:
            ```python
            # In your call() method:
            temperature = self._get_parameter("temperature", kwargs, model)
            max_tokens = self._get_parameter("max_tokens", kwargs, model)
            
            # This will use kwargs["temperature"] if provided, otherwise
            # the default value from the configuration file
            ```
        
        Note:
            This method automatically logs when kwargs override config values
            for debugging purposes.
        """
        config_value = self.config_loader.get_parameter_default(self.provider_name, model, param_name)
        if param_name in kwargs:
            logger.debug(f"Overriding {param_name} from config ({config_value}) with kwargs ({kwargs[param_name]})")
            return kwargs[param_name]
        return config_value

    def _convert_parameter_type(self, param_name: str, value: Any, model: str) -> Any:
        """
        Convert parameter value to the correct type based on configuration.
        
        This method performs automatic type conversion for parameters based on the
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
        
        Raises:
            ValueError: If conversion fails or unknown type is specified
        
        Example:
            ```python
            # Convert various parameter types
            temp = self._convert_parameter_type("temperature", "0.7", model)  # → 0.7 (float)
            max_tokens = self._convert_parameter_type("max_tokens", "1000", model)  # → 1000 (int)
            use_search = self._convert_parameter_type("use_websearch", "true", model)  # → True (bool)
            stop_words = self._convert_parameter_type("stop_sequences", '["END", "STOP"]', model)  # → ["END", "STOP"] (list)
            ```
        
        Note:
            This method is typically called after `_get_parameter()` to ensure
            all parameters have the correct type before API calls.
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
        Make a call to the LLM provider with the given prompt and parameters.
        
        This is the main interface method that users will call to interact with
        the LLM provider. It handles parameter resolution, type conversion,
        API communication, and response parsing.
        
        Implementation Guidelines:
        1. **Parameter Resolution**: Use `_get_parameter()` to get parameter values
        2. **Type Conversion**: Use `_convert_parameter_type()` for proper typing
        3. **API Communication**: Call `_make_request()` for the actual API call
        4. **Response Parsing**: Use `_parse_response()` to convert to LLMResponse
        5. **Error Handling**: Implement appropriate error handling and logging
        6. **Latency Tracking**: Measure and include latency in the response
        
        Args:
            prompt (str): The user prompt/query to send to the LLM
            system_prompt (Optional[str]): Optional system prompt that sets the
                context or behavior for the LLM. If None, no system prompt is used.
            **kwargs: Additional parameters that can override configuration defaults.
                Common parameters include:
                - `model` (str): Model identifier to use
                - `temperature` (float): Sampling temperature (0.0-2.0)
                - `max_tokens` (int): Maximum tokens to generate
                - `top_p` (float): Nucleus sampling parameter
                - `stop` (List[str]): Stop sequences
                - Provider-specific parameters as defined in config
        
        Returns:
            LLMResponse: Standardized response object containing:
                - `text` (str): Generated text response
                - `provider` (str): Provider name
                - `model` (str): Model used
                - `cost` (float): Estimated cost in USD (if calculable)
                - `input_tokens` (int): Number of input tokens
                - `output_tokens` (int): Number of output tokens
                - `latency_ms` (float): Response latency in milliseconds
                - `metadata` (Dict): Additional provider-specific metadata
        
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
        
        Raises:
            Various exceptions may be raised depending on the provider implementation.
            Common exceptions include:
            - API authentication errors
            - Rate limiting errors  
            - Invalid parameter errors
            - Network connectivity errors
        """
        pass

    @abstractmethod
    def _make_request(self, **kwargs) -> Any:
        """
        Make the actual API request to the provider's service.
        
        This method handles the low-level communication with the provider's API.
        It should be implemented to make the actual HTTP/API call and return
        the raw response object from the provider.
        
        Implementation Guidelines:
        1. **API Client**: Use the API client initialized in `__init__`
        2. **Parameter Mapping**: Map Octuner parameters to provider-specific API parameters
        3. **Error Handling**: Handle API-specific errors and convert to appropriate exceptions
        4. **Retry Logic**: Implement retry logic for transient failures if needed
        5. **Logging**: Log API calls and responses for debugging
        
        Args:
            **kwargs: Request parameters that have been processed and type-converted.
                Common parameters include:
                - `model` (str): Model identifier
                - `prompt` (str): User prompt
                - `system_prompt` (str): System prompt (if supported)
                - `temperature` (float): Sampling temperature
                - `max_tokens` (int): Maximum tokens to generate
                - `top_p` (float): Nucleus sampling parameter
                - Provider-specific parameters
        
        Returns:
            Any: Raw response object from the provider's API. This will be passed
                to `_parse_response()` for conversion to LLMResponse.
        
        Example Implementation:
            ```python
            def _make_request(self, **kwargs) -> Any:
                # Example for a REST API
                response = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": kwargs["model"],
                        "messages": [
                            {"role": "system", "content": kwargs.get("system_prompt")},
                            {"role": "user", "content": kwargs["prompt"]}
                        ],
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 1000)
                    }
                )
                return response.json()
            ```
        
        Raises:
            Various exceptions may be raised depending on the provider:
            - `requests.exceptions.RequestException`: For HTTP-related errors
            - `openai.APIError`: For OpenAI-specific errors
            - `google.generativeai.types.GoogleAPIError`: For Gemini-specific errors
            - Custom provider-specific exceptions
        """
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> LLMResponse:
        """
        Parse the raw API response into a standardized LLMResponse object.
        
        This method converts the provider-specific API response into Octuner's
        standardized LLMResponse format. It extracts text content, metadata,
        and token usage information from the raw response.
        
        Implementation Guidelines:
        1. **Text Extraction**: Extract the generated text from the response
        2. **Token Counting**: Extract input and output token counts if available
        3. **Metadata**: Include relevant response metadata
        4. **Error Handling**: Handle cases where expected fields are missing
        5. **Provider Info**: Set provider name and model information
        
        Args:
            response (Any): Raw response object from the provider's API,
                as returned by `_make_request()`
        
        Returns:
            LLMResponse: Standardized response object with:
                - `text` (str): Generated text content
                - `provider` (str): Provider name (should match self.provider_name)
                - `model` (str): Model identifier used
                - `cost` (float): Estimated cost in USD (if calculable)
                - `input_tokens` (int): Number of input tokens consumed
                - `output_tokens` (int): Number of output tokens generated
                - `latency_ms` (float): Response latency (if measured)
                - `metadata` (Dict): Additional provider-specific metadata
        
        Example Implementation:
            ```python
            def _parse_response(self, response: Any) -> LLMResponse:
                # Extract text content
                text = ""
                if hasattr(response, 'choices') and response.choices:
                    text = response.choices[0].message.content or ""
                
                # Extract token usage
                input_tokens = None
                output_tokens = None
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                
                # Calculate cost if possible
                cost = None
                if input_tokens and output_tokens:
                    cost = self._calculate_cost(input_tokens, output_tokens, response.model)
                
                return LLMResponse(
                    text=text,
                    provider=self.provider_name,
                    model=response.model,
                    cost=cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=None,  # Set in call() method
                    metadata={
                        "response_id": getattr(response, 'id', None),
                        "created": getattr(response, 'created', None),
                        "finish_reason": getattr(response.choices[0], 'finish_reason', None) if response.choices else None
                    }
                )
            ```
        
        Note:
            - The `latency_ms` field should be set in the `call()` method, not here
            - Cost calculation should use `_calculate_cost()` if token counts are available
            - Handle missing fields gracefully with appropriate defaults
        """
        pass

    @abstractmethod
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """
        Calculate the cost for a given number of input and output tokens.
        
        This method calculates the estimated cost of an API call based on the
        number of tokens consumed and the pricing information from the configuration.
        The cost calculation is essential for optimization algorithms that consider
        cost as a factor in their decision-making.
        
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
            float: Total cost in USD for the API call
        
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
        
        Note:
            - Pricing information should be defined in the configuration file
            - If pricing is not available, return 0.0 or None
            - The framework uses this for cost-aware optimization
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


# Example Implementation
"""
Complete example of implementing a custom LLM provider:

```python
import time
import requests
from typing import Any, Dict, List, Optional, Tuple
from octuner.providers.base import BaseLLMProvider, LLMResponse
from octuner.config.loader import ConfigLoader

class CustomLLMProvider(BaseLLMProvider):
    \"\"\"
    Example implementation of a custom LLM provider.
    
    This example shows how to implement a provider for a hypothetical
    REST API-based LLM service.
    \"\"\"
    
    def __init__(self, config_loader, **kwargs):
        super().__init__(config_loader, **kwargs)
        self.provider_name = "custom"
        
        # Initialize API client
        self.api_key = kwargs.get('api_key')
        self.base_url = kwargs.get('base_url', 'https://api.custom-llm.com')
        self.timeout = kwargs.get('timeout', 30)
        
        if not self.api_key:
            raise ValueError("api_key is required for CustomLLMProvider")
    
    def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        \"\"\"Make a call to the custom LLM API.\"\"\"
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
        
        # Parse response
        result = self._parse_response(response)
        result.latency_ms = (time.time() - start_time) * 1000
        return result
    
    def _make_request(self, **kwargs) -> Any:
        \"\"\"Make the actual API request.\"\"\"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=kwargs,
            headers=headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _parse_response(self, response: Any) -> LLMResponse:
        \"\"\"Parse the API response.\"\"\"
        # Extract text content
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract token usage
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        
        # Calculate cost
        cost = None
        if input_tokens and output_tokens:
            cost = self._calculate_cost(input_tokens, output_tokens, response.get("model"))
        
        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=response.get("model"),
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=None,  # Set in call() method
            metadata={
                "response_id": response.get("id"),
                "created": response.get("created"),
                "finish_reason": response.get("choices", [{}])[0].get("finish_reason")
            }
        )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        \"\"\"Calculate cost based on token usage.\"\"\"
        input_cost_per_1m, output_cost_per_1m = self.get_cost_per_token(model)
        input_cost = (input_tokens * input_cost_per_1m) / 1_000_000
        output_cost = (output_tokens * output_cost_per_1m) / 1_000_000
        return input_cost + output_cost

# Configuration file (custom_config.yaml):
\"\"\"
providers:
  custom:
    default_model: "custom-model-v1"
    available_models: ["custom-model-v1", "custom-model-v2"]
    pricing_usd_per_1m_tokens:
      custom-model-v1: [0.5, 1.0]  # [input_cost, output_cost]
      custom-model-v2: [1.0, 2.0]
    model_capabilities:
      custom-model-v1:
        supported_parameters: ["temperature", "max_tokens", "top_p"]
        parameters:
          temperature:
            type: float
            range: [0.0, 2.0]
            default: 0.7
          max_tokens:
            type: int
            range: [1, 4000]
            default: 1000
          top_p:
            type: float
            range: [0.0, 1.0]
            default: 0.9
        forced_parameters: {}
\"\"\"

# Usage:
# config_loader = ConfigLoader("custom_config.yaml")
# provider = CustomLLMProvider(config_loader, api_key="your-key")
# response = provider.call("Hello, world!")
```
"""