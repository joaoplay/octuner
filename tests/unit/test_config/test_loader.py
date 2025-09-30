"""
Unit tests for ConfigLoader functionality.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from octuner.config.loader import ConfigLoader


class TestConfigLoader:
    """Test ConfigLoader class functionality."""

    def test_init_with_valid_config(self, config_file):
        """Test ConfigLoader initialization with valid config file."""
        loader = ConfigLoader(config_file)
        assert loader.config_file == Path(config_file)
        assert loader._config is not None

    def test_init_with_nonexistent_file(self):
        """Test ConfigLoader initialization with non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ConfigLoader("nonexistent_file.yaml")
        assert "Configuration file not found" in str(exc_info.value)

    def test_init_with_invalid_yaml(self, invalid_config_file):
        """Test ConfigLoader initialization with invalid YAML."""
        with pytest.raises(ValueError) as exc_info:
            ConfigLoader(invalid_config_file)
        assert "Invalid YAML" in str(exc_info.value)

    def test_get_providers(self, config_loader):
        """Test getting list of providers."""
        providers = config_loader.get_providers()
        assert providers == ["openai"]

    def test_get_providers_empty_config(self, temp_dir):
        """Test getting providers from empty config."""
        config_path = temp_dir / "empty_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump({}, f)
        
        loader = ConfigLoader(str(config_path))
        providers = loader.get_providers()
        assert providers == []

    def test_get_provider_config_valid(self, config_loader):
        """Test getting valid provider configuration."""
        provider_config = config_loader.get_provider_config("openai")
        assert "api_key_env" in provider_config
        assert "default_model" in provider_config
        assert provider_config["default_model"] == "gpt-4o-mini"

    def test_get_provider_config_invalid(self, config_loader):
        """Test getting invalid provider configuration."""
        with pytest.raises(ValueError) as exc_info:
            config_loader.get_provider_config("nonexistent")
        assert "Provider 'nonexistent' not found" in str(exc_info.value)

    def test_get_default_model(self, config_loader):
        """Test getting default model for provider."""
        default_model = config_loader.get_default_model("openai")
        assert default_model == "gpt-4o-mini"

    def test_get_default_model_missing(self, temp_dir):
        """Test getting default model when not specified."""
        config_data = {
            "providers": {
                "test_provider": {
                    "api_key_env": "TEST_KEY"
                    # Missing default_model
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.get_default_model("test_provider")
        assert "No default_model specified" in str(exc_info.value)

    def test_get_available_models(self, config_loader):
        """Test getting available models for provider."""
        models = config_loader.get_available_models("openai")
        expected_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
        assert models == expected_models

    def test_get_available_models_empty(self, temp_dir):
        """Test getting available models when not specified."""
        config_data = {
            "providers": {
                "test_provider": {
                    "default_model": "test-model"
                    # Missing available_models
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        models = loader.get_available_models("test_provider")
        assert models == []

    def test_get_pricing(self, config_loader):
        """Test getting pricing information."""
        input_cost, output_cost = config_loader.get_pricing("openai", "gpt-4o-mini")
        assert input_cost == 0.15
        assert output_cost == 0.6

    def test_get_pricing_fallback(self, config_loader):
        """Test getting pricing with fallback for unknown model."""
        with patch('octuner.config.loader.logger') as mock_logger:
            input_cost, output_cost = config_loader.get_pricing("openai", "unknown-model")
            assert input_cost == 0.15  # fallback
            assert output_cost == 0.60  # fallback
            mock_logger.warning.assert_called_once()

    def test_get_model_capabilities(self, config_loader):
        """Test getting model capabilities."""
        capabilities = config_loader.get_model_capabilities("openai", "gpt-4o-mini")
        assert "supported_parameters" in capabilities
        assert "parameters" in capabilities
        assert "temperature" in capabilities["parameters"]

    def test_get_model_capabilities_missing(self, config_loader):
        """Test getting capabilities for non-existent model."""
        with pytest.raises(ValueError) as exc_info:
            config_loader.get_model_capabilities("openai", "nonexistent-model")
        assert "Model 'nonexistent-model' not found" in str(exc_info.value)

    def test_get_supported_parameters(self, config_loader):
        """Test getting supported parameters for model."""
        params = config_loader.get_supported_parameters("openai", "gpt-4o-mini")
        expected_params = ["temperature", "top_p", "max_tokens", "frequency_penalty"]
        assert params == expected_params

    def test_model_supports_parameter(self, config_loader):
        """Test checking if model supports parameter."""
        assert config_loader.model_supports_parameter("openai", "gpt-4o-mini", "temperature")
        assert not config_loader.model_supports_parameter("openai", "gpt-4o-mini", "unsupported_param")

    def test_get_parameter_range_float(self, config_loader):
        """Test getting parameter range for float parameter."""
        range_values = config_loader.get_parameter_range("openai", "gpt-4o-mini", "temperature")
        assert range_values == (0.0, 2.0)

    def test_get_parameter_range_int(self, config_loader):
        """Test getting parameter range for int parameter."""
        range_values = config_loader.get_parameter_range("openai", "gpt-4o-mini", "max_tokens")
        assert range_values == (50, 4000)

    def test_get_parameter_range_special_websearch(self, config_loader):
        """Test getting parameter range for special websearch parameter."""
        range_values = config_loader.get_parameter_range("openai", "gpt-4o-mini", "use_websearch")
        assert range_values == (True, False)

    def test_get_parameter_range_special_search_context(self, config_loader):
        """Test getting parameter range for special search context size parameter."""
        range_values = config_loader.get_parameter_range("openai", "gpt-4o-mini", "search_context_size")
        assert range_values == (1, 20)

    def test_get_parameter_range_missing(self, config_loader):
        """Test getting parameter range for non-existent parameter."""
        with pytest.raises(ValueError) as exc_info:
            config_loader.get_parameter_range("openai", "gpt-4o-mini", "nonexistent_param")
        assert "No parameter definition found" in str(exc_info.value)

    def test_get_parameter_default(self, config_loader):
        """Test getting parameter default value."""
        default = config_loader.get_parameter_default("openai", "gpt-4o-mini", "temperature")
        assert default == 0.7

    def test_get_parameter_default_model_special(self, config_loader):
        """Test getting default for special 'model' parameter."""
        default = config_loader.get_parameter_default("openai", "gpt-4o-mini", "model")
        assert default == "gpt-4o-mini"

    def test_get_parameter_default_missing(self, temp_dir):
        """Test getting parameter default when not defined."""
        config_data = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            "parameters": {
                                "temperature": {
                                    "type": "float",
                                    "range": [0.0, 2.0]
                                    # Missing default
                                }
                            }
                        }
                    }
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.get_parameter_default("openai", "gpt-4o-mini", "temperature")
        assert "No default value defined" in str(exc_info.value)

    def test_get_parameter_type(self, config_loader):
        """Test getting parameter type."""
        param_type = config_loader.get_parameter_type("openai", "gpt-4o-mini", "temperature")
        assert param_type == "float"

        param_type = config_loader.get_parameter_type("openai", "gpt-4o-mini", "max_tokens")
        assert param_type == "int"

    def test_get_parameter_type_model_special(self, config_loader):
        """Test getting type for special 'model' parameter."""
        param_type = config_loader.get_parameter_type("openai", "gpt-4o-mini", "model")
        assert param_type == "str"

    def test_get_parameter_type_missing(self, temp_dir):
        """Test getting parameter type when not defined."""
        config_data = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            "parameters": {
                                "temperature": {
                                    "range": [0.0, 2.0]
                                    # Missing type
                                }
                            }
                        }
                    }
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.get_parameter_type("openai", "gpt-4o-mini", "temperature")
        assert "Parameter type for 'temperature' not defined" in str(exc_info.value)

    def test_get_forced_parameter(self, temp_dir):
        """Test getting forced parameter value."""
        config_data = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            "forced_parameters": {
                                "max_tokens": 500
                            }
                        }
                    }
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        forced_value = loader.get_forced_parameter("openai", "gpt-4o-mini", "max_tokens")
        assert forced_value == 500

        # Test non-forced parameter
        forced_value = loader.get_forced_parameter("openai", "gpt-4o-mini", "temperature")
        assert forced_value is None


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_validate_config_valid(self, config_loader):
        """Test validation of valid configuration."""
        assert config_loader.validate_config() is True

    def test_validate_config_missing_providers(self, temp_dir):
        """Test validation with missing providers section."""
        config_data = {"invalid": "config"}
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.validate_config()
        assert "Configuration must have 'providers' section" in str(exc_info.value)

    def test_validate_config_missing_default_model(self, temp_dir):
        """Test validation with missing default_model."""
        config_data = {
            "providers": {
                "openai": {
                    "api_key_env": "OPENAI_API_KEY"
                    # Missing default_model
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.validate_config()
        assert "must have 'default_model'" in str(exc_info.value)

    def test_validate_config_missing_supported_parameters(self, temp_dir):
        """Test validation with missing supported_parameters."""
        config_data = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            # Missing supported_parameters
                            "parameters": {}
                        }
                    }
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.validate_config()
        assert "must have 'supported_parameters'" in str(exc_info.value)

    def test_validate_config_missing_parameter_type(self, temp_dir):
        """Test validation with missing parameter type."""
        config_data = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            "supported_parameters": ["temperature"],
                            "parameters": {
                                "temperature": {
                                    "range": [0.0, 2.0]
                                    # Missing type
                                }
                            }
                        }
                    }
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.validate_config()
        assert "must have 'type' defined" in str(exc_info.value)

    def test_validate_config_missing_parameter_range(self, temp_dir):
        """Test validation with missing parameter range."""
        config_data = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            "supported_parameters": ["temperature"],
                            "parameters": {
                                "temperature": {
                                    "type": "float"
                                    # Missing range
                                }
                            }
                        }
                    }
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        with pytest.raises(ValueError) as exc_info:
            loader.validate_config()
        assert "must have 'range' defined" in str(exc_info.value)

    @patch('octuner.config.loader.logger')
    def test_validate_config_warns_missing_default(self, mock_logger, temp_dir):
        """Test validation warns about missing default values."""
        config_data = {
            "providers": {
                "openai": {
                    "default_model": "gpt-4o-mini",
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            "supported_parameters": ["temperature"],
                            "parameters": {
                                "temperature": {
                                    "type": "float",
                                    "range": [0.0, 2.0]
                                    # Missing default (should warn but not fail)
                                }
                            }
                        }
                    }
                }
            }
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_path))
        assert loader.validate_config() is True
        mock_logger.warning.assert_called()
