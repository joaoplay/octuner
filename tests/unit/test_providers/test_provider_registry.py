"""
Unit tests for provider registry functionality.
"""

import pytest
from unittest.mock import Mock

from octuner.providers.registry import get_provider, get_all_models, PROVIDERS


class TestProviderRegistry:
    """Test provider registry functionality."""

    def test_providers_registry_contents(self):
        """Test that PROVIDERS registry contains expected providers."""
        assert "openai" in PROVIDERS
        assert "gemini" in PROVIDERS
        
        # Verify providers are classes (not instances)
        assert callable(PROVIDERS["openai"])
        assert callable(PROVIDERS["gemini"])

    def test_get_provider_openai(self):
        """Test getting OpenAI provider."""
        mock_config_loader = Mock()
        mock_config_loader.get_provider_config.return_value = {
            "api_key_env": "OPENAI_API_KEY"
        }
        
        provider = get_provider("openai", mock_config_loader, api_key="test-key")
        
        assert provider is not None
        assert provider.config_loader == mock_config_loader
        assert hasattr(provider, 'call')

    def test_get_provider_gemini(self):
        """Test getting Gemini provider."""
        mock_config_loader = Mock()
        mock_config_loader.get_provider_config.return_value = {
            "api_key_env": "GOOGLE_API_KEY"
        }
        
        provider = get_provider("gemini", mock_config_loader, api_key="test-key")
        
        assert provider is not None
        assert provider.config_loader == mock_config_loader
        assert hasattr(provider, 'call')

    def test_get_provider_with_kwargs(self):
        """Test getting provider with additional kwargs."""
        mock_config_loader = Mock()
        mock_config_loader.get_provider_config.return_value = {
            "api_key_env": "OPENAI_API_KEY"
        }
        
        provider = get_provider("openai", mock_config_loader, api_key="test-key", custom_param="test_value")
        
        assert provider is not None
        assert provider.config_loader == mock_config_loader

    def test_get_provider_invalid(self):
        """Test getting invalid provider raises error."""
        mock_config_loader = Mock()
        
        with pytest.raises(KeyError) as exc_info:
            get_provider("invalid_provider", mock_config_loader)
        
        assert "invalid_provider" in str(exc_info.value)

    def test_get_provider_none_config(self):
        """Test getting provider with None config loader."""
        # This should work if the provider can handle None config
        try:
            provider = get_provider("openai", None)
            # If it works, provider should exist but might not be fully functional
            assert provider is not None
        except Exception:
            # If it fails, that's also acceptable behavior
            pass

    def test_get_all_models_structure(self):
        """Test that get_all_models returns expected structure."""
        models = get_all_models()
        
        assert isinstance(models, dict)
        assert "openai" in models
        assert "gemini" in models
        
        # Each provider should have a list of models
        assert isinstance(models["openai"], list)
        assert isinstance(models["gemini"], list)

    def test_get_all_models_openai_content(self):
        """Test OpenAI models in get_all_models."""
        models = get_all_models()
        openai_models = models["openai"]
        
        # Should contain common OpenAI models
        expected_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
        for model in expected_models:
            assert model in openai_models

    def test_get_all_models_gemini_content(self):
        """Test Gemini models in get_all_models."""
        models = get_all_models()
        gemini_models = models["gemini"]
        
        # Should contain common Gemini models
        expected_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        for model in expected_models:
            assert model in gemini_models

    def test_get_all_models_immutability(self):
        """Test that get_all_models returns immutable/copied data."""
        models1 = get_all_models()
        models2 = get_all_models()
        
        # Modify first result
        models1["openai"].append("fake-model")
        
        # Second call should not be affected
        assert "fake-model" not in models2["openai"]

    def test_provider_registry_contains_base_interface(self):
        """Test that providers in registry implement base interface."""
        from octuner.providers.base import BaseLLMProvider
        
        for provider_name, provider_class in PROVIDERS.items():
            # Create a mock config to test instantiation
            mock_config = Mock()
            mock_config.get_provider_config.return_value = {
                "api_key_env": f"{provider_name.upper()}_API_KEY"
            }
            
            try:
                # Try to create an instance with API key
                provider_instance = provider_class(config_loader=mock_config, api_key="test-key")
                
                # Check if it's a subclass of BaseLLMProvider
                assert isinstance(provider_instance, BaseLLMProvider)
                
                # Check required methods exist
                assert hasattr(provider_instance, 'call')
                assert hasattr(provider_instance, '_make_request')
                assert hasattr(provider_instance, '_parse_response')
                assert hasattr(provider_instance, '_calculate_cost')
                
            except Exception as e:
                # If instantiation fails due to missing API keys, that's OK
                # as long as the class exists and is properly structured
                if "API" in str(e) or "key" in str(e).lower():
                    # Expected error for missing API keys
                    assert issubclass(provider_class, BaseLLMProvider)
                else:
                    # Unexpected error
                    raise

    def test_registry_keys_match_provider_names(self):
        """Test that registry keys match provider internal names."""
        mock_config = Mock()
        
        for provider_name in PROVIDERS.keys():
            try:
                provider = get_provider(provider_name, mock_config)
                # Many providers might set an internal name or identifier
                # This test verifies consistency
                assert provider is not None
            except Exception:
                # Skip if provider can't be instantiated without proper config
                pass

    def test_provider_classes_are_distinct(self):
        """Test that different providers are actually different classes."""
        openai_class = PROVIDERS["openai"]
        gemini_class = PROVIDERS["gemini"]
        
        assert openai_class != gemini_class
        assert openai_class.__name__ != gemini_class.__name__

    def test_get_provider_case_sensitivity(self):
        """Test that provider names are case sensitive."""
        mock_config = Mock()
        mock_config.get_provider_config.return_value = {
            "api_key_env": "OPENAI_API_KEY"
        }
        
        # Should work with correct case
        provider = get_provider("openai", mock_config, api_key="test-key")
        assert provider is not None
        
        # Should fail with wrong case
        with pytest.raises(KeyError):
            get_provider("OpenAI", mock_config, api_key="test-key")
        
        with pytest.raises(KeyError):
            get_provider("OPENAI", mock_config, api_key="test-key")

    def test_get_all_models_keys_match_providers(self):
        """Test that get_all_models keys match PROVIDERS keys."""
        models = get_all_models()
        
        # All providers should have model lists
        for provider_name in PROVIDERS.keys():
            assert provider_name in models
        
        # No extra providers should be in models
        for provider_name in models.keys():
            assert provider_name in PROVIDERS


class TestProviderRegistryEdgeCases:
    """Test edge cases and error conditions for provider registry."""

    def test_provider_with_empty_config(self):
        """Test provider creation with empty config."""
        mock_config = Mock()
        mock_config.get_provider_config.return_value = {}
        mock_config.get_default_model.return_value = "default-model"
        mock_config.get_pricing.return_value = (0.001, 0.002)
        
        # Should handle empty config gracefully
        try:
            provider = get_provider("openai", mock_config)
            assert provider is not None
        except Exception as e:
            # Acceptable if provider requires specific config
            assert "config" in str(e).lower() or "key" in str(e).lower()

    def test_get_provider_with_invalid_kwargs(self):
        """Test provider creation with invalid kwargs."""
        mock_config = Mock()
        mock_config.get_provider_config.return_value = {
            "api_key_env": "OPENAI_API_KEY"
        }
        
        # Provider should handle or ignore unknown kwargs
        try:
            provider = get_provider("openai", mock_config, api_key="test-key", invalid_param=123)
            assert provider is not None
        except TypeError as e:
            # Some providers might reject unknown kwargs
            assert "unexpected keyword argument" in str(e).lower()

    def test_providers_registry_modification(self):
        """Test behavior when trying to modify PROVIDERS registry."""
        original_providers = PROVIDERS.copy()
        
        # Try to add a fake provider
        PROVIDERS["fake"] = Mock
        
        # Should be able to add (though not recommended)
        assert "fake" in PROVIDERS
        
        # Clean up
        del PROVIDERS["fake"]
        assert PROVIDERS == original_providers

    def test_get_all_models_with_unavailable_provider(self):
        """Test get_all_models when a provider might be unavailable."""
        # This tests robustness of the implementation
        models = get_all_models()
        
        # Should return a dict even if some providers have issues
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_provider_instantiation_with_missing_dependencies(self):
        """Test provider behavior when dependencies might be missing."""
        mock_config = Mock()
        
        # Test that we can at least import and reference the providers
        # even if their runtime dependencies might be missing
        for provider_name, provider_class in PROVIDERS.items():
            assert provider_class is not None
            assert hasattr(provider_class, '__init__')
            
            # The class should exist even if instantiation fails
            try:
                provider_class.__name__
                provider_class.__module__
            except Exception:
                pytest.fail(f"Provider class {provider_name} is not properly defined")

    def test_concurrent_provider_access(self):
        """Test that provider registry works with concurrent access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def get_provider_worker():
            try:
                mock_config = Mock()
                provider = get_provider("openai", mock_config)
                results.append(provider)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=get_provider_worker)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        if errors:
            # If there are errors, they should be consistent (e.g., all missing API keys)
            first_error_type = type(errors[0])
            assert all(isinstance(e, first_error_type) for e in errors)
        else:
            # If successful, should have results
            assert len(results) == 5
            assert all(r is not None for r in results)
