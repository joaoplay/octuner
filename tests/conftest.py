"""
Shared pytest fixtures and configuration for Octuner tests.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import octuner
from octuner.config.loader import ConfigLoader
from octuner.tunable.types import DatasetItem, MetricResult, TrialResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_openai_config():
    """Sample OpenAI configuration for testing."""
    return {
        "providers": {
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
                "available_models": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
                "pricing_usd_per_1m_tokens": {
                    "gpt-3.5-turbo": [0.5, 1.5],
                    "gpt-4o": [5.0, 15.0],
                    "gpt-4o-mini": [0.15, 0.6]
                },
                "model_capabilities": {
                    "gpt-4o-mini": {
                        "supported_parameters": ["temperature", "top_p", "max_tokens", "frequency_penalty"],
                        "parameters": {
                            "temperature": {
                                "type": "float",
                                "range": [0.0, 2.0],
                                "default": 0.7
                            },
                            "top_p": {
                                "type": "float",
                                "range": [0.0, 1.0],
                                "default": 1.0
                            },
                            "max_tokens": {
                                "type": "int",
                                "range": [50, 4000],
                                "default": 1000
                            },
                            "frequency_penalty": {
                                "type": "float",
                                "range": [0.0, 2.0],
                                "default": 0.0
                            }
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_gemini_config():
    """Sample Gemini configuration for testing."""
    return {
        "providers": {
            "gemini": {
                "api_key_env": "GOOGLE_API_KEY",
                "default_model": "gemini-1.5-flash",
                "available_models": ["gemini-1.5-flash", "gemini-1.5-pro"],
                "pricing_usd_per_1m_tokens": {
                    "gemini-1.5-flash": [0.075, 0.3],
                    "gemini-1.5-pro": [1.25, 5.0]
                },
                "model_capabilities": {
                    "gemini-1.5-flash": {
                        "supported_parameters": ["temperature", "top_p", "max_tokens", "use_websearch"],
                        "parameters": {
                            "temperature": {
                                "type": "float",
                                "range": [0.0, 2.0],
                                "default": 0.7
                            },
                            "top_p": {
                                "type": "float",
                                "range": [0.0, 1.0],
                                "default": 0.95
                            },
                            "max_tokens": {
                                "type": "int",
                                "range": [1, 8192],
                                "default": 1000
                            }
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_multi_provider_config(sample_openai_config, sample_gemini_config):
    """Sample multi-provider configuration combining OpenAI and Gemini."""
    config = {"providers": {}}
    config["providers"].update(sample_openai_config["providers"])
    config["providers"].update(sample_gemini_config["providers"])
    return config


@pytest.fixture
def config_file(temp_dir, sample_openai_config):
    """Create a temporary config file for testing."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_openai_config, f)
    return str(config_path)


@pytest.fixture
def invalid_config_file(temp_dir):
    """Create an invalid config file for error testing."""
    config_path = temp_dir / "invalid_config.yaml"
    with open(config_path, 'w') as f:
        f.write("invalid: yaml: content: [unclosed")
    return str(config_path)


@pytest.fixture
def config_loader(config_file):
    """Create a ConfigLoader instance with test configuration."""
    return ConfigLoader(config_file)


@pytest.fixture
def sample_dataset():
    """Sample dataset for optimization testing."""
    return [
        {"input": "I love this product!", "target": {"sentiment": "positive", "confidence": 0.9}},
        {"input": "This is terrible", "target": {"sentiment": "negative", "confidence": 0.8}},
        {"input": "It's okay", "target": {"sentiment": "neutral", "confidence": 0.6}},
        {"input": "Amazing quality!", "target": {"sentiment": "positive", "confidence": 0.95}},
        {"input": "Worst purchase ever", "target": {"sentiment": "negative", "confidence": 0.9}}
    ]


@pytest.fixture
def sample_trial_results():
    """Sample trial results for testing."""
    return [
        TrialResult(
            trial_number=1,
            parameters={"temperature": 0.5, "top_p": 0.9},
            metrics=MetricResult(quality=0.85, cost=0.002, latency_ms=120),
            success=True
        ),
        TrialResult(
            trial_number=2,
            parameters={"temperature": 0.7, "top_p": 0.8},
            metrics=MetricResult(quality=0.78, cost=0.003, latency_ms=150),
            success=True
        ),
        TrialResult(
            trial_number=3,
            parameters={"temperature": 1.0, "top_p": 0.95},
            metrics=MetricResult(quality=0.65, cost=0.004, latency_ms=180),
            success=False,
            error="Token limit exceeded"
        )
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1699900000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from OpenAI."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "This is a test response from Gemini."}
                    ],
                    "role": "model"
                },
                "finish_reason": "STOP",
                "index": 0
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 8,
            "candidates_token_count": 12,
            "total_token_count": 20
        }
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    with patch('google.generativeai.configure') as mock_configure, \
         patch('google.generativeai.GenerativeModel') as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_optuna_study():
    """Mock Optuna study for optimization testing."""
    with patch('optuna.create_study') as mock_create_study:
        mock_study = Mock()
        mock_create_study.return_value = mock_study
        yield mock_study


@pytest.fixture(autouse=True)
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key',
        'GOOGLE_SEARCH_API_KEY': 'test-search-key',
        'GOOGLE_SEARCH_ENGINE_ID': 'test-engine-id'
    }
    
    with patch.dict('os.environ', env_vars, clear=False):
        yield env_vars


class MockLLMResponse:
    """Mock LLM response for testing."""
    def __init__(self, text: str, provider: str = "test", model: str = "test-model", 
                 cost: float = 0.001, latency_ms: float = 100):
        self.text = text
        self.provider = provider
        self.model = model
        self.cost = cost
        self.latency_ms = latency_ms
        self.metadata = {
            "tokens_used": {"input": 10, "output": 15, "total": 25}
        }


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return MockLLMResponse("Test response from mock LLM")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services (skipped by default)"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "fixtures"


# Helper functions for tests
def create_test_config_file(temp_dir: Path, config_data: Dict[str, Any], filename: str = "config.yaml") -> str:
    """Helper to create a test configuration file."""
    config_path = temp_dir / filename
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return str(config_path)


def assert_config_validation_error(config_data: Dict[str, Any], expected_error_message: str = None):
    """Helper to assert that config validation raises appropriate errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        if expected_error_message:
            with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
                ConfigLoader(str(config_path))
            assert expected_error_message in str(exc_info.value)
        else:
            with pytest.raises((ValueError, FileNotFoundError)):
                ConfigLoader(str(config_path))
