import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Import OpenAI for patching in tests
import openai
OpenAI = openai.OpenAI


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation for Octuner.

    This module contains the OpenAI provider implementation using the Responses API.
    """

    def __init__(self, config_loader, **kwargs):
        import openai
        super().__init__(config_loader=config_loader, **kwargs)
        self.provider_name = "openai"

        # Get timeout from config or kwargs
        config_timeout = self.config_loader.get_provider_config(self.provider_name).get('timeout', 120)
        timeout = kwargs.get('timeout', config_timeout)

        # Get API key from kwargs or environment variable
        api_key = kwargs.get('api_key')
        if not api_key:
            import os
            provider_config = config_loader.get_provider_config(self.provider_name)
            api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.environ.get(api_key_env)

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=kwargs.get('base_url'),
            timeout=timeout
        )


    def call(self, prompt: str, system_prompt: Optional[str] = None, use_websearch: bool = False, **kwargs) -> LLMResponse:
        """
        Make a call to OpenAI Chat Completions API, with optional websearch support.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            use_websearch: Whether to enable websearch tool
            **kwargs: Additional parameters to override defaults (e.g., model, temperature, max_tokens, etc.)
        """
        # Measure start time for latency tracking
        start_time = time.time()

        # Get the model to use
        model = self._get_parameter("model", kwargs, "gpt-4o-mini")

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare base request
        api_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if use_websearch:
            # Configure web search tool
            tool = {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
            api_params["tools"] = [tool]

        # All the remaining parameters are dynamically added if supported by the model. The config_loader
        # knows which parameters are supported for each model. See the YAML config files.
        supported_params = self.config_loader.get_supported_parameters(self.provider_name, model)

        for param in supported_params:
            param_value = self._get_parameter(param, kwargs, model)
            if param_value is None:
                continue

            # Convert the value to the correct type if needed
            converted_value = self._convert_parameter_type(param, param_value, model)
            
            # Map parameter names to what OpenAI API expects
            if param == "max_output_tokens":
                api_param_name = "max_tokens"
            else:
                api_param_name = param
            
            api_params[api_param_name] = converted_value

        # Make the API call with retries on timeout
        # FIXME: Make the number of retries configurable
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                break
            except Exception as e:
                if "timeout" in str(e).lower() and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Timeout (attempt {attempt+1}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise

        latency_ms = (time.time() - start_time) * 1000

        # Parse response text
        text = ""
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content or ""

        # Calculate cost if possible
        cost = None
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._calculate_cost(input_tokens, output_tokens, model)
        else:
            input_tokens = None
            output_tokens = None

        return LLMResponse(
            text=text,
            provider="openai",
            model=model,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            metadata={
                "model": response.model,
                "response_id": getattr(response, 'id', None),
                "created": getattr(response, 'created', None),
                "tokens_used": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": (input_tokens or 0) + (output_tokens or 0)
                }
            },
        )

    def _make_request(self, **kwargs) -> Any:
        """
        Make the actual API request to OpenAI.
        
        Args:
            **kwargs: Request parameters
            
        Returns:
            Raw response from OpenAI API
        """
        return self.client.chat.completions.create(**kwargs)

    def _parse_response(self, response: Any) -> LLMResponse:
        """
        Parse the raw OpenAI response into an LLMResponse.
        
        Args:
            response: Raw response from OpenAI API
            
        Returns:
            Parsed LLMResponse
        """
        # Parse response text
        text = ""
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content or ""

        # Get token usage
        input_tokens = None
        output_tokens = None
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        # Calculate cost if possible
        cost = None
        if input_tokens is not None and output_tokens is not None:
            try:
                cost = self._calculate_cost(input_tokens, output_tokens, response.model)
            except Exception:
                cost = None

        return LLMResponse(
            text=text,
            provider="openai",
            model=response.model,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=None,
            metadata={
                "model": response.model,
                "response_id": getattr(response, 'id', None),
                "created": getattr(response, 'created', None),
                "tokens_used": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": (input_tokens or 0) + (output_tokens or 0)
                }
            },
        )

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
        input_cost, output_cost = self.get_cost_per_token(model)
        return (input_tokens * input_cost / 1_000_000) + (output_tokens * output_cost / 1_000_000)

