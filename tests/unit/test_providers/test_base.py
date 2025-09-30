"""
Unit tests for BaseLLMProvider interface.
"""

import pytest
from abc import ABC
from unittest.mock import Mock

from octuner.providers.base import BaseLLMProvider, LLMResponse


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_llm_response_basic(self):
        """Test basic LLMResponse creation."""
        response = LLMResponse(
            text="Hello, world!",
            provider="test_provider",
            model="test_model"
        )
        
        assert response.text == "Hello, world!"
        assert response.provider == "test_provider"
        assert response.model == "test_model"
        assert response.cost is None
        assert response.latency_ms is None
        assert response.metadata is None

    def test_llm_response_with_all_fields(self):
        """Test LLMResponse with all optional fields."""
        metadata = {
            "tokens_used": {"input": 10, "output": 15, "total": 25},
            "finish_reason": "stop"
        }
        
        response = LLMResponse(
            text="Complete response",
            provider="openai",
            model="gpt-4o-mini",
            cost=0.002,
            latency_ms=150.5,
            metadata=metadata
        )
        
        assert response.text == "Complete response"
        assert response.provider == "openai"
        assert response.model == "gpt-4o-mini"
        assert response.cost == 0.002
        assert response.latency_ms == 150.5
        assert response.metadata == metadata
        assert response.metadata["tokens_used"]["total"] == 25

    def test_llm_response_empty_text(self):
        """Test LLMResponse with empty text."""
        response = LLMResponse(
            text="",
            provider="test",
            model="test-model"
        )
        
        assert response.text == ""
        assert response.provider == "test"
        assert response.model == "test-model"

    def test_llm_response_zero_cost(self):
        """Test LLMResponse with zero cost."""
        response = LLMResponse(
            text="Free response",
            provider="test",
            model="free-model",
            cost=0.0,
            latency_ms=0.0
        )
        
        assert response.cost == 0.0
        assert response.latency_ms == 0.0

    def test_llm_response_negative_values(self):
        """Test LLMResponse with negative values (edge case)."""
        response = LLMResponse(
            text="Response",
            provider="test",
            model="test",
            cost=-0.001,  # Could happen with credits
            latency_ms=-10.0  # Edge case
        )
        
        assert response.cost == -0.001
        assert response.latency_ms == -10.0

    def test_llm_response_metadata_immutability(self):
        """Test that metadata is properly stored (not enforced immutable)."""
        original_metadata = {"key": "value"}
        response = LLMResponse(
            text="Test",
            provider="test",
            model="test",
            metadata=original_metadata
        )
        
        # Modify original metadata
        original_metadata["key"] = "modified"
        
        # Response metadata should be affected (no deep copy)
        assert response.metadata["key"] == "modified"


class TestBaseLLMProvider:
    """Test BaseLLMProvider abstract base class."""

    def test_base_provider_is_abstract(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_base_provider_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Check that the required abstract methods exist
        abstract_methods = BaseLLMProvider.__abstractmethods__
        expected_methods = {'call', '_make_request', '_parse_response', '_calculate_cost'}
        
        assert expected_methods.issubset(abstract_methods)

    def test_concrete_implementation_basic(self):
        """Test that concrete implementation can be created."""
        class ConcreteLLMProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
            
            def call(self, message, **kwargs):
                return LLMResponse("test response", "test", "test-model")
            
            def _make_request(self, messages, **kwargs):
                return {"response": "test"}
            
            def _parse_response(self, response):
                return "test response"
            
            def _calculate_cost(self, input_tokens, output_tokens):
                return 0.001
        
        mock_config = Mock()
        provider = ConcreteLLMProvider(mock_config)
        
        assert provider.config_loader == mock_config
        assert isinstance(provider, BaseLLMProvider)

    def test_concrete_implementation_call_method(self):
        """Test that concrete implementation call method works."""
        class TestProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
            
            def call(self, message, **kwargs):
                return LLMResponse(
                    text=f"Response to: {message}",
                    provider="test",
                    model="test-model"
                )
            
            def _make_request(self, messages, **kwargs):
                return {"text": f"Response to: {messages}"}
            
            def _parse_response(self, response):
                return response["text"]
            
            def _calculate_cost(self, input_tokens, output_tokens):
                return input_tokens * 0.001 + output_tokens * 0.002
        
        mock_config = Mock()
        provider = TestProvider(mock_config)
        
        response = provider.call("Hello")
        assert isinstance(response, LLMResponse)
        assert response.text == "Response to: Hello"
        assert response.provider == "test"
        assert response.model == "test-model"

    def test_missing_abstract_method_implementation(self):
        """Test that missing abstract method implementation raises error."""
        class IncompleteProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
            
            # Missing implementations of call, _make_request, _parse_response, _calculate_cost
            pass
        
        mock_config = Mock()
        with pytest.raises(TypeError):
            IncompleteProvider(mock_config)

    def test_partial_abstract_method_implementation(self):
        """Test that partial implementation of abstract methods raises error."""
        class PartialProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
            
            def call(self, message, **kwargs):
                return LLMResponse("test", "test", "test")
            
            def _make_request(self, messages, **kwargs):
                return {}
            
            # Missing _parse_response and _calculate_cost
        
        mock_config = Mock()
        with pytest.raises(TypeError):
            PartialProvider(mock_config)

    def test_provider_inheritance_hierarchy(self):
        """Test that provider inheritance hierarchy works correctly."""
        class BaseCustomProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
                self.base_attribute = "base"
            
            def call(self, message, **kwargs):
                return LLMResponse("base response", "base", "base-model")
            
            def _make_request(self, messages, **kwargs):
                return {"response": "base"}
            
            def _parse_response(self, response):
                return response["response"]
            
            def _calculate_cost(self, input_tokens, output_tokens):
                return 0.001
        
        class ExtendedProvider(BaseCustomProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
                self.extended_attribute = "extended"
            
            def call(self, message, **kwargs):
                base_response = super().call(message, **kwargs)
                return LLMResponse(
                    f"Extended: {base_response.text}",
                    "extended",
                    "extended-model"
                )
        
        mock_config = Mock()
        
        # Now the class should work correctly
        provider = BaseCustomProvider(mock_config)
        assert provider.base_attribute == "base"

    def test_provider_inheritance_hierarchy_fixed(self):
        """Test that provider inheritance hierarchy works correctly (fixed version)."""
        class BaseCustomProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
                self.base_attribute = "base"
            
            def call(self, message, **kwargs):
                return LLMResponse("base response", "base", "base-model")
            
            def _make_request(self, messages, **kwargs):
                return {"response": "base"}
            
            def _parse_response(self, response):
                return response["response"]
            
            def _calculate_cost(self, input_tokens, output_tokens):
                return 0.001
        
        class ExtendedProvider(BaseCustomProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
                self.extended_attribute = "extended"
            
            def call(self, message, **kwargs):
                base_response = super().call(message, **kwargs)
                return LLMResponse(
                    f"Extended: {base_response.text}",
                    "extended",
                    "extended-model"
                )
        
        mock_config = Mock()
        
        base_provider = BaseCustomProvider(mock_config)
        extended_provider = ExtendedProvider(mock_config)
        
        assert base_provider.base_attribute == "base"
        assert extended_provider.base_attribute == "base"
        assert extended_provider.extended_attribute == "extended"
        
        base_response = base_provider.call("test")
        extended_response = extended_provider.call("test")
        
        assert base_response.text == "base response"
        assert extended_response.text == "Extended: base response"


class TestProviderWithDifferentSignatures:
    """Test providers with different method signatures."""

    def test_provider_with_additional_methods(self):
        """Test provider with additional methods beyond abstract interface."""
        class AdvancedProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
            
            def call(self, message, **kwargs):
                return LLMResponse("response", "advanced", "advanced-model")
            
            def _make_request(self, messages, **kwargs):
                return {"text": "response"}
            
            def _parse_response(self, response):
                return response["text"]
            
            def _calculate_cost(self, input_tokens, output_tokens):
                return 0.001
            
            # Additional methods
            def batch_call(self, messages):
                """Batch processing method."""
                return [self.call(msg) for msg in messages]
            
            def stream_call(self, message):
                """Streaming method."""
                for chunk in ["Hello", " ", "World"]:
                    yield LLMResponse(chunk, "advanced", "advanced-model")
        
        mock_config = Mock()
        provider = AdvancedProvider(mock_config)
        
        # Test basic functionality
        response = provider.call("test")
        assert response.text == "response"
        
        # Test additional methods
        batch_responses = provider.batch_call(["msg1", "msg2"])
        assert len(batch_responses) == 2
        
        stream_responses = list(provider.stream_call("test"))
        assert len(stream_responses) == 3
        assert stream_responses[0].text == "Hello"

    def test_provider_with_different_call_signature(self):
        """Test provider with extended call method signature."""
        class CustomSignatureProvider(BaseLLMProvider):
            def __init__(self, config_loader):
                super().__init__(config_loader)
            
            def call(self, message, system_role=None, temperature=0.7, **kwargs):
                response_text = f"System: {system_role}, Temp: {temperature}, Message: {message}"
                return LLMResponse(response_text, "custom", "custom-model")
            
            def _make_request(self, messages, **kwargs):
                return {"text": "response"}
            
            def _parse_response(self, response):
                return response["text"]
            
            def _calculate_cost(self, input_tokens, output_tokens):
                return 0.001
        
        mock_config = Mock()
        provider = CustomSignatureProvider(mock_config)
        
        # Test with default parameters
        response1 = provider.call("Hello")
        assert "Temp: 0.7" in response1.text
        assert "System: None" in response1.text
        
        # Test with custom parameters
        response2 = provider.call("Hello", system_role="assistant", temperature=1.0)
        assert "Temp: 1.0" in response2.text
        assert "System: assistant" in response2.text
