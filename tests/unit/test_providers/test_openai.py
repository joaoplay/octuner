"""
Unit tests for OpenAI provider with mocked API calls.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI

from octuner.providers.openai import OpenAIProvider
from octuner.providers.base import LLMResponse


class TestOpenAIProvider:
    """Test OpenAI provider functionality with mocked API calls."""

    @pytest.fixture
    def mock_config_loader(self):
        """Create a mock config loader for OpenAI."""
        mock_config = Mock()
        mock_config.get_provider_config.return_value = {
            "api_key_env": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini"
        }
        mock_config.get_default_model.return_value = "gpt-4o-mini"
        mock_config.get_pricing.return_value = (0.15, 0.6)  # input, output per 1M tokens
        mock_config.get_supported_parameters.return_value = ["temperature", "max_output_tokens"]
        mock_config.get_parameter_default.side_effect = lambda provider, model, param: {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_output_tokens": 1000
        }.get(param)
        mock_config.get_parameter_type.side_effect = lambda provider, model, param: {
            "temperature": "float",
            "max_output_tokens": "int"
        }.get(param, "str")
        mock_config.get_parameter_range.side_effect = lambda provider, model, param: {
            "temperature": [0.0, 2.0],
            "max_output_tokens": [1, 4000]
        }.get(param, [])
        return mock_config

    @patch('openai.OpenAI')
    def test_provider_initialization(self, mock_openai_class, mock_config_loader):
        """Test OpenAI provider initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        assert provider.config_loader == mock_config_loader
        assert provider.client == mock_client
        mock_openai_class.assert_called_once()

    @patch('openai.OpenAI')
    def test_call_basic(self, mock_openai_class, mock_config_loader):
        """Test basic call to OpenAI API."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock the API response for chat.completions.create()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response from OpenAI"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15)
        mock_response.model = "gpt-4o-mini"

        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        start_time = time.time()
        response = provider.call("Hello, world!")
        end_time = time.time()
        
        assert isinstance(response, LLMResponse)
        assert response.text == "Test response from OpenAI"
        assert response.provider == "openai"
        assert response.model == "gpt-4o-mini"
        assert response.latency_ms > 0  # Should measure latency
        assert response.metadata is not None

    @patch('openai.OpenAI')
    def test_call_with_parameters(self, mock_openai_class, mock_config_loader):
        """Test call with custom parameters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Custom response"))]
        mock_response.usage = Mock(prompt_tokens=8, completion_tokens=12)
        mock_response.model = "gpt-4o-mini"

        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        response = provider.call(
            "Hello",
            temperature=0.8,
            max_output_tokens=500
        )
        
        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["max_tokens"] == 500
        assert call_args[1]["model"] is not None  # Model should be set

    @patch('openai.OpenAI')
    def test_call_with_system_role(self, mock_openai_class, mock_config_loader):
        """Test call with system prompt."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Assistant response"))]
        mock_response.usage = Mock(prompt_tokens=15, completion_tokens=10)
        mock_response.model = "gpt-4o-mini"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        response = provider.call("Hello", system_prompt="You are a helpful assistant.")
        
        # Verify the API was called
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        
        # Check that messages include system prompt
        messages = call_args[1]["messages"]
        assert any(msg["role"] == "system" for msg in messages)

    @patch('openai.OpenAI')
    def test_call_with_websearch(self, mock_openai_class, mock_config_loader):
        """Test call with web search enabled."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Web search response"))]
        mock_response.usage = Mock(prompt_tokens=20, completion_tokens=30)
        mock_response.model = "gpt-4o-mini"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        response = provider.call("Current weather", use_websearch=True)
        
        # Verify web search tool was included
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        tools = call_args[1].get("tools", [])
        
        assert len(tools) > 0
        assert any(tool["type"] == "function" for tool in tools)

    @patch('openai.OpenAI')
    def test_make_request_basic(self, mock_openai_class, mock_config_loader):
        """Test _make_request method."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        result = provider._make_request(model="gpt-4o-mini", messages=[{"role": "user", "content": "Test message"}], temperature=0.7)
        
        assert result == mock_response
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_parse_response(self, mock_openai_class, mock_config_loader):
        """Test _parse_response method."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        # Test successful response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Parsed content"))]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15)
        
        parsed = provider._parse_response(mock_response)
        assert isinstance(parsed, LLMResponse)
        assert parsed.text == "Parsed content"

    @patch('openai.OpenAI')
    def test_parse_response_empty_choices(self, mock_openai_class, mock_config_loader):
        """Test _parse_response with empty output."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=0)
        
        parsed = provider._parse_response(mock_response)
        assert isinstance(parsed, LLMResponse)
        assert parsed.text == ""

    @patch('openai.OpenAI')
    def test_calculate_cost(self, mock_openai_class, mock_config_loader):
        """Test _calculate_cost method."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        # Test cost calculation
        cost = provider._calculate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o-mini")
        
        # Expected: (1000 * 0.15 + 500 * 0.6) / 1_000_000 = 0.00045
        expected_cost = (1000 * 0.15 + 500 * 0.6) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

    @patch('openai.OpenAI')
    def test_calculate_cost_zero_tokens(self, mock_openai_class, mock_config_loader):
        """Test cost calculation with zero tokens."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        cost = provider._calculate_cost(input_tokens=0, output_tokens=0, model="gpt-4o-mini")
        assert cost == 0.0

    @patch('openai.OpenAI')
    def test_error_handling_api_error(self, mock_openai_class, mock_config_loader):
        """Test error handling for API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock API error - use generic Exception since OpenAI API structure may vary
        mock_client.responses.create.side_effect = Exception("API Error")
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        with pytest.raises(Exception):
            provider.call("Test message")

    @patch('openai.OpenAI')
    def test_error_handling_rate_limit(self, mock_openai_class, mock_config_loader):
        """Test error handling for rate limit errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock rate limit error - use generic Exception with specific message
        mock_client.responses.create.side_effect = Exception("Rate limit exceeded")
        
        provider = OpenAIProvider(config_loader=mock_config_loader, api_key="test-api-key")
        
        with pytest.raises(Exception):
            provider.call("Test message")

    @patch('openai.OpenAI')
    def test_different_models(self, mock_openai_class, mock_config_loader):
        """Test provider with different models."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Model response"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15, total_tokens=25)
        mock_response.model = "gpt-4o"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader)
        
        response = provider.call("Hello", model="gpt-4o")
        
        # Verify correct model was used
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o"
        assert response.model == "gpt-4o"

    @patch('openai.OpenAI')
    def test_metadata_tracking(self, mock_openai_class, mock_config_loader):
        """Test that response metadata is properly tracked."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=12, completion_tokens=18, total_tokens=30)
        mock_response.model = "gpt-4o-mini"
        mock_response.id = "chatcmpl-test123"
        mock_response.created = 1699900000
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader)
        
        response = provider.call("Test")
        
        assert response.metadata is not None
        assert response.metadata["tokens_used"]["input"] == 12
        assert response.metadata["tokens_used"]["output"] == 18
        assert response.metadata["tokens_used"]["total"] == 30
        assert response.metadata["response_id"] == "chatcmpl-test123"
        assert response.metadata["created"] == 1699900000

    @patch('openai.OpenAI')
    def test_latency_measurement(self, mock_openai_class, mock_config_loader):
        """Test that latency is properly measured."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        mock_response.model = "gpt-4o-mini"
        
        # Add delay to simulate API latency
        def delayed_response(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return mock_response
        
        mock_client.chat.completions.create.side_effect = delayed_response
        
        provider = OpenAIProvider(config_loader=mock_config_loader)
        
        response = provider.call("Test")
        
        # Should measure at least 100ms of latency
        assert response.latency_ms >= 100
        assert response.latency_ms < 200  # Should be less than 200ms for test
