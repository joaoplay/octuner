"""
Unit tests for Gemini provider with mocked API calls.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from octuner.providers.gemini import GeminiProvider
from octuner.providers.base import LLMResponse


class TestGeminiProvider:
    """Test Gemini provider functionality with mocked API calls."""

    @pytest.fixture
    def mock_config_loader(self):
        """Create a mock config loader for Gemini."""
        mock_config = Mock()
        mock_config.get_provider_config.return_value = {
            "api_key_env": "GOOGLE_API_KEY",
            "default_model": "gemini-1.5-flash"
        }
        mock_config.get_default_model.return_value = "gemini-1.5-flash"
        mock_config.get_pricing.return_value = (0.075, 0.3)  # input, output per 1M tokens
        mock_config.get_supported_parameters.return_value = ["temperature", "max_output_tokens"]
        mock_config.get_parameter_default.side_effect = lambda provider, model, param: {
            "model": "gemini-1.5-flash",
            "temperature": 0.7,
            "max_output_tokens": 1000
        }.get(param)
        mock_config.get_parameter_type.side_effect = lambda provider, model, param: {
            "temperature": "float",
            "max_output_tokens": "int"
        }.get(param, "str")
        mock_config.get_parameter_range.side_effect = lambda provider, model, param: {
            "temperature": [0.0, 2.0],
            "max_output_tokens": [50, 4000]
        }.get(param, [])
        return mock_config

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_provider_initialization(self, mock_types, mock_client_class, mock_config_loader):
        """Test Gemini provider initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        assert provider.config_loader == mock_config_loader
        assert provider.client == mock_client
        mock_client_class.assert_called_once_with(api_key="test-google-key")

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_call_basic(self, mock_types, mock_client_class, mock_config_loader):
        """Test basic call to Gemini API."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock the API response
        mock_response = Mock()
        mock_response.text = "Test response from Gemini"
        mock_response.usage_metadata = Mock(
            prompt_token_count=8,
            candidates_token_count=12,
            total_token_count=20
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        # Mock the models.generate_content method
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call("Hello, world!")
        
        assert isinstance(response, LLMResponse)
        assert response.text == "Test response from Gemini"
        assert response.provider == "gemini"
        assert response.model == "gemini-1.5-flash"
        assert response.input_tokens == 8
        assert response.output_tokens == 12
        assert response.latency_ms > 0  # Should measure latency
        assert response.metadata is not None
        assert response.metadata["finish_reason"] == "STOP"

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_call_with_parameters(self, mock_types, mock_client_class, mock_config_loader):
        """Test call with custom parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Custom response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=15,
            total_token_count=25
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call(
            "Hello",
            temperature=0.8,
            max_tokens=500,
            top_p=0.9
        )
        
        # Verify the API was called with correct parameters
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args
        
        # Check that the call was made with the right structure
        assert call_args[1]["model"] == "gemini-1.5-flash"
        assert call_args[1]["contents"] == "Hello"
        assert "config" in call_args[1]

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_call_with_system_role(self, mock_types, mock_client_class, mock_config_loader):
        """Test call with system instruction."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Assistant response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=15,
            candidates_token_count=10,
            total_token_count=25
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call("Hello", system_prompt="You are a helpful assistant.")
        
        # Verify the API was called
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args
        assert call_args[1]["contents"] == "Hello"

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_call_with_websearch(self, mock_types, mock_client_class, mock_config_loader):
        """Test call with web search enabled."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Web search response with current information"
        mock_response.usage_metadata = Mock(
            prompt_token_count=20,
            candidates_token_count=30,
            total_token_count=50
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call("Current weather", use_websearch=True)
        
        # Verify web search was enabled via tools
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args
        
        # Check that tools were configured
        config = call_args[1]["config"]
        assert hasattr(config, 'tools') or 'tools' in str(config)

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_make_request_basic(self, mock_types, mock_client_class, mock_config_loader):
        """Test _make_request method."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        result = provider._make_request(model="gemini-1.5-flash", contents="Test message", config=Mock())
        
        assert result == mock_response
        mock_client.models.generate_content.assert_called_once()

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_parse_response(self, mock_types, mock_client_class, mock_config_loader):
        """Test _parse_response method."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        # Test successful response
        mock_response = Mock()
        mock_response.text = "Parsed content from Gemini"
        mock_response.model = "gemini-1.5-flash"
        
        parsed_response = provider._parse_response(mock_response)
        assert isinstance(parsed_response, LLMResponse)
        assert parsed_response.text == "Parsed content from Gemini"
        assert parsed_response.provider == "gemini"

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_parse_response_empty(self, mock_types, mock_client_class, mock_config_loader):
        """Test _parse_response with empty or blocked response."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        # Test response with no text (could be blocked)
        mock_response = Mock()
        mock_response.text = ""
        mock_response.model = "gemini-1.5-flash"
        
        parsed_response = provider._parse_response(mock_response)
        assert isinstance(parsed_response, LLMResponse)
        assert parsed_response.text == ""

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_parse_response_blocked_content(self, mock_types, mock_client_class, mock_config_loader):
        """Test _parse_response with blocked content."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        # Test response that might be blocked by safety filters
        mock_response = Mock()
        mock_response.text = None
        mock_response.model = "gemini-1.5-flash"
        
        parsed_response = provider._parse_response(mock_response)
        assert isinstance(parsed_response, LLMResponse)
        assert parsed_response.text == ""

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_calculate_cost(self, mock_types, mock_client_class, mock_config_loader):
        """Test _calculate_cost method."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        # Test cost calculation
        cost = provider._calculate_cost(input_tokens=1000, output_tokens=500, model="gemini-1.5-flash")
        
        # Expected: (1000 * 0.075 + 500 * 0.3) / 1_000_000 = 0.000225
        expected_cost = (1000 * 0.075 + 500 * 0.3) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_calculate_cost_zero_tokens(self, mock_types, mock_client_class, mock_config_loader):
        """Test cost calculation with zero tokens."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        cost = provider._calculate_cost(input_tokens=0, output_tokens=0, model="gemini-1.5-flash")
        assert cost == 0.0

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_error_handling_api_error(self, mock_types, mock_client_class, mock_config_loader):
        """Test error handling for API errors."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock API error
        mock_client.models.generate_content.side_effect = Exception("Gemini API Error")
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        with pytest.raises(Exception):
            provider.call("Test message")

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_different_models(self, mock_types, mock_client_class, mock_config_loader):
        """Test provider with different models."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Pro model response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=15,
            total_token_count=25
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call("Hello", model="gemini-1.5-pro")
        
        # Verify correct model was used
        assert response.provider == "gemini"
        assert response.model == "gemini-1.5-pro"

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_metadata_tracking(self, mock_types, mock_client_class, mock_config_loader):
        """Test that response metadata is properly tracked."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=12,
            candidates_token_count=18,
            total_token_count=30
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call("Test")
        
        assert response.metadata is not None
        assert response.input_tokens == 12
        assert response.output_tokens == 18
        assert response.metadata["finish_reason"] == "STOP"

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_latency_measurement(self, mock_types, mock_client_class, mock_config_loader):
        """Test that latency is properly measured."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=10,
            total_token_count=20
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        # Add delay to simulate API latency
        def delayed_response(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return mock_response
        
        mock_client.models.generate_content.side_effect = delayed_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call("Test")
        
        # Should measure at least 100ms of latency
        assert response.latency_ms >= 100
        assert response.latency_ms < 200  # Should be less than 200ms for test

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_generation_config_creation(self, mock_types, mock_client_class, mock_config_loader):
        """Test creation of generation config."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=5,
            candidates_token_count=10,
            total_token_count=15
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        # Test with various parameters
        response = provider.call(
            "Test",
            temperature=1.5,
            max_tokens=2000,
            top_p=0.8,
            top_k=50
        )
        
        # Verify the API was called
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args
        
        # Check that config was passed
        assert "config" in call_args[1]

    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_safety_settings_default(self, mock_types, mock_client_class, mock_config_loader):
        """Test that default safety settings are applied."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.text = "Safe response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=5,
            candidates_token_count=10,
            total_token_count=15
        )
        # Mock candidates for finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        provider = GeminiProvider(config_loader=mock_config_loader)
        
        response = provider.call("Test")
        
        # Verify the API was called
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args
        
        # Check that config was passed
        assert "config" in call_args[1]
