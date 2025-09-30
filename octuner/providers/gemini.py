import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Gemini provider implementation. Using the google-generativeai SDK.
    """

    def __init__(self, config_loader, **kwargs):
        super().__init__(config_loader=config_loader, **kwargs)
        self.provider_name = "gemini"

        from google import genai
        from google.genai import types
        
        # Get API key from kwargs or environment variable
        api_key = kwargs.get("api_key")
        if not api_key:
            import os
            provider_config = config_loader.get_provider_config(self.provider_name)
            api_key_env = provider_config.get("api_key_env", "GOOGLE_API_KEY")
            api_key = os.environ.get(api_key_env)
        
        self.client = genai.Client(api_key=api_key)
        self.types = types

    def call(self, prompt: str, system_prompt: Optional[str] = None, use_websearch: bool = False, **kwargs) -> LLMResponse:
        """
        Make a call to Gemini API, with optional websearch support.
        """
        start_time = time.time()
        model = self._get_parameter("model", kwargs, "gemini-1.5-flash")

        config_params = {}
        supported_params = self.config_loader.get_supported_parameters(self.provider_name, model)

        # Parameters that are not valid for GenerateContentConfig
        invalid_params = {
            'use_websearch', 'search_context_size', 'user_location',
            'model', 'system_prompt', 'prompt'
        }

        for param in supported_params:
            # Skip parameters that are not valid for GenerateContentConfig
            if param in invalid_params:
                logger.debug(f"Skipping invalid parameter for GenerateContentConfig: {param}")
                continue

            param_value = self._get_parameter(param, kwargs, model)
            if param_value is None:
                continue
            converted_value = self._convert_parameter_type(param, param_value, model)

            # Map parameter name to Gemini API expected name
            config_params[param] = converted_value
            logger.debug(f"Added parameter {param} = {converted_value}")


        tools = None
        if use_websearch:
            # Create GoogleSearch tool for web search capabilities using new API
            grounding_tool = self.types.Tool(
                google_search=self.types.GoogleSearch()
            )
            tools = [grounding_tool]
            logger.debug("Added GoogleSearch tool for web search capabilities")

        # Only pass valid parameters to GenerateContentConfig
        valid_config_params = {}
        for key, value in config_params.items():
            # Additional validation for known valid parameters
            if key in ['temperature', 'max_output_tokens', 'top_p', 'top_k',
                      'candidate_count', 'stop_sequences']:
                valid_config_params[key] = value
            else:
                logger.debug(f"Skipping unknown parameter for GenerateContentConfig: {key}")

        logger.debug(f"Creating GenerateContentConfig with parameters: {valid_config_params}")
        config = self.types.GenerateContentConfig(**valid_config_params)
        if tools:
            config.tools = tools
            logger.debug(f"Added tools to config: {tools}")

        # -----------------------------
        # Execute API call using new client structure
        # -----------------------------
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        latency_ms = (time.time() - start_time) * 1000
        text = getattr(response, "text", None) or ""

        # Extract grounding metadata if web search was used
        grounding_metadata = None
        if use_websearch and hasattr(response, 'grounding_metadata'):
            grounding_metadata = response.grounding_metadata
            logger.debug(f"Grounding metadata available: {grounding_metadata}")

        # Extract usage metadata
        input_tokens = None
        output_tokens = None
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            input_tokens = getattr(usage, 'prompt_token_count', None)
            output_tokens = getattr(usage, 'candidates_token_count', None)

        # Extract finish reason
        finish_reason = None
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)

        return LLMResponse(
            text=text,
            provider="gemini",
            model=model,
            cost=None,  # Cost calculation would need to be implemented
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            metadata={
                "model": model,
                "finish_reason": finish_reason,
                "grounding_metadata": grounding_metadata,
                "websearch_enabled": use_websearch,
            },
        )

    def _make_request(self, **kwargs) -> Any:
        """
        Make the actual API request to Gemini.
        
        Args:
            **kwargs: Request parameters
            
        Returns:
            Raw response from Gemini API
        """
        return self.client.models.generate_content(**kwargs)

    def _parse_response(self, response: Any) -> LLMResponse:
        """
        Parse the raw Gemini response into an LLMResponse.
        
        Args:
            response: Raw response from Gemini API
            
        Returns:
            Parsed LLMResponse
        """
        text = ""
        if hasattr(response, "text"):
            text = response.text or ""
        elif hasattr(response, "candidates") and response.candidates:
            text = response.candidates[0].content.parts[0].text or ""

        return LLMResponse(
            text=text,
            provider="gemini",
            model=getattr(response, "model", None),
            cost=None,
            input_tokens=None,
            output_tokens=None,
            latency_ms=None,
            metadata={"model": getattr(response, "model", None)},
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
