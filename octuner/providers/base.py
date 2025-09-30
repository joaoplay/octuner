"""
Base provider interface and common functionality for Octuner.

This module contains the abstract base class and common types for LLM providers.
"""

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
    Abstract base class for LLM providers.
    
    All provider implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(self, config_loader, **kwargs):
        """
        Initialize the provider with configuration.
        
        Args:
            config_loader: ConfigLoader for configuration-driven behavior (mandatory)
            **kwargs: Provider-specific configuration
        """
        if config_loader is None:
            raise ValueError("config_loader is mandatory for all providers")
        
        self.config = kwargs
        self.config_loader = config_loader
        self.provider_name = getattr(self, 'provider_name', None)  # Set by subclasses

    def _get_parameter(self, param_name: str, kwargs: Dict[str, Any], model: str) -> Any:
        """Get parameter from config loader, with optional kwargs override."""
        config_value = self.config_loader.get_parameter_default(self.provider_name, model, param_name)
        if param_name in kwargs:
            logger.debug(f"Overriding {param_name} from config ({config_value}) with kwargs ({kwargs[param_name]})")
            return kwargs[param_name]
        return config_value

    def _convert_parameter_type(self, param_name: str, value: Any, model: str) -> Any:
        """
        Convert parameter value to correct type based on config.
        
        This method handles type conversion for parameters based on the expected type
        defined in the configuration. Supports int, float, str, bool, choice, and list types.
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
        Make a call to the LLM provider.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse with the result
        """
        pass

    @abstractmethod
    def _make_request(self, **kwargs) -> Any:
        """
        Make the actual API request to the provider.
        
        Args:
            **kwargs: Request parameters
            
        Returns:
            Raw response from the provider API
        """
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> LLMResponse:
        """
        Parse the raw API response into an LLMResponse.
        
        Args:
            response: Raw response from the provider API
            
        Returns:
            Parsed LLMResponse
        """
        pass

    @abstractmethod
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """
        Calculate the cost for a given number of tokens.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier
            
        Returns:
            Cost in USD
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