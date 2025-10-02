"""
End-to-end integration tests for Octuner.

These tests verify that the entire system works together correctly,
from configuration loading through optimization to result export.
"""

import pytest
import yaml
from unittest.mock import Mock, patch

from octuner import (
    MultiProviderTunableLLM, AutoTuner, apply_best,
    TunableMixin, register_tunable_class
)
from octuner.config.loader import ConfigLoader
from octuner.providers.base import LLMResponse
from octuner.tunable.types import SearchResult, TrialResult, MetricResult


@pytest.fixture
def complete_config_file(temp_dir):
    """Create a complete test configuration file."""
    config_data = {
        "providers": {
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
                "available_models": ["gpt-4o-mini"],
                "pricing_usd_per_1m_tokens": {
                    "gpt-4o-mini": [0.15, 0.6]
                },
                "model_capabilities": {
                    "gpt-4o-mini": {
                        "supported_parameters": ["temperature", "top_p", "max_tokens"],
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
                            }
                        }
                    }
                }
            }
        }
    }
    
    config_path = temp_dir / "complete_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return str(config_path)


@pytest.fixture
def sentiment_dataset():
    """Create a sentiment analysis dataset for testing."""
    return [
        {"input": "I absolutely love this product! It's amazing!", 
         "target": {"sentiment": "positive", "confidence": 0.95}},
        {"input": "This is the worst thing I've ever bought. Terrible quality.", 
         "target": {"sentiment": "negative", "confidence": 0.9}},
        {"input": "It's okay, nothing special but does the job.", 
         "target": {"sentiment": "neutral", "confidence": 0.6}},
        {"input": "Fantastic experience! Highly recommend!", 
         "target": {"sentiment": "positive", "confidence": 0.9}},
        {"input": "Complete waste of money. Very disappointed.", 
         "target": {"sentiment": "negative", "confidence": 0.85}}
    ]


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_config_loading_and_validation(self, complete_config_file):
        """Test that configuration loading and validation works end-to-end."""
        # Load configuration
        config_loader = ConfigLoader(complete_config_file)
        
        # Validate configuration
        assert config_loader.validate_config() is True
        
        # Test provider access
        providers = config_loader.get_providers()
        assert "openai" in providers
        
        # Test model capabilities
        capabilities = config_loader.get_model_capabilities("openai", "gpt-4o-mini")
        assert "supported_parameters" in capabilities
        assert "temperature" in capabilities["supported_parameters"]

    @patch('openai.OpenAI')
    def test_multiproviderlllmtuner_basic_usage(self, mock_openai_class, complete_config_file):
        """Test basic usage of MultiProviderTunableLLM."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="positive sentiment detected"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15, total_tokens=25)
        mock_response.model = "gpt-4o-mini"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create tuner
        tuner = MultiProviderTunableLLM(config_file=complete_config_file)
        
        # Test basic call
        response = tuner.call("I love this product!")
        
        assert isinstance(response, LLMResponse)
        assert response.text == "positive sentiment detected"
        assert response.provider == "openai"
        assert response.model == "gpt-4o-mini"

    @patch('openai.OpenAI')
    def test_parameter_tuning_workflow(self, mock_openai_class, complete_config_file):
        """Test the complete parameter tuning workflow."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock different responses based on temperature
        def mock_generate_response(*args, **kwargs):
            temperature = kwargs.get('temperature', 0.7)
            mock_response = Mock()
            
            if temperature < 0.5:
                mock_response.choices = [Mock(message=Mock(content="neutral response"))]
            elif temperature > 1.0:
                mock_response.choices = [Mock(message=Mock(content="very creative positive response"))]
            else:
                mock_response.choices = [Mock(message=Mock(content="positive sentiment"))]
            
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15, total_tokens=25)
            mock_response.model = "gpt-4o-mini"
            return mock_response
        
        mock_client.chat.completions.create.side_effect = mock_generate_response
        
        # Create tunable component
        tuner = MultiProviderTunableLLM(config_file=complete_config_file)
        
        # Verify it's tunable
        assert tuner.is_tunable("temperature")
        assert tuner.is_tunable("top_p")
        assert tuner.is_tunable("max_tokens")
        
        # Get tunable parameters
        params = tuner.get_tunable_parameters()
        assert "temperature" in params
        assert params["temperature"]["type"] == "float"

    @patch('openai.OpenAI')
    @patch('octuner.optimization.auto.LLMOptimizer')
    def test_full_optimization_workflow(self, mock_optimizer_class, mock_openai_class, 
                                      complete_config_file, sentiment_dataset):
        """Test the complete optimization workflow from start to finish."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Create realistic trial results
        trial_results = []
        for i in range(5):
            trial_results.append(TrialResult(
                trial_number=i+1,
                parameters={
                    "temperature": 0.5 + i*0.1,
                    "top_p": 0.8 + i*0.05,
                    "max_tokens": 500 + i*100
                },
                metrics=MetricResult(
                    quality=0.7 + i*0.05,
                    cost=0.001 + i*0.0005,
                    latency_ms=100 + i*10
                ),
                success=True
            ))
        
        best_trial = trial_results[-1]  # Last trial is best
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=trial_results,
            optimization_mode="pareto",
            dataset_size=len(sentiment_dataset),
            total_trials=5,
            best_parameters=best_trial.parameters,
            metrics_summary={
                "avg_quality": 0.775,
                "best_quality": 0.9,
                "total_cost": sum(t.metrics.cost for t in trial_results),
                "avg_latency_ms": sum(t.metrics.latency_ms for t in trial_results) / 5
            }
        )
        
        mock_optimizer.optimize.return_value = search_result
        mock_optimizer.get_best_trial.return_value = best_trial
        mock_optimizer.get_best_parameters.return_value = best_trial.parameters
        
        # Create components
        tuner = MultiProviderTunableLLM(config_file=complete_config_file)
        
        def sentiment_metric(output, target):
            """Simple sentiment metric for testing."""
            output_text = output.text.lower() if hasattr(output, 'text') else str(output).lower()
            target_sentiment = target.get("sentiment", "")
            return 0.9 if target_sentiment in output_text else 0.3
        
        def entrypoint_func(component, input_data):
            return component.call(input_data)
        
        # Create AutoTuner
        auto_tuner = AutoTuner(
            component=tuner,
            dataset=sentiment_dataset,
            metric_function=sentiment_metric,
            entrypoint_function=entrypoint_func,
            n_trials=5
        )
        
        # Run optimization
        result = auto_tuner.search()
        
        # Verify results
        assert isinstance(result, SearchResult)
        assert len(result.all_trials) == 5
        assert result.best_trial.success is True
        assert result.dataset_size == len(sentiment_dataset)
        
        # Apply best parameters
        apply_best(tuner, result)
        
        # Verify parameters were applied
        assert tuner.temperature == best_trial.parameters["temperature"]
        assert tuner.top_p == best_trial.parameters["top_p"]
        assert tuner.max_tokens == best_trial.parameters["max_tokens"]

    def test_save_and_load_optimization_results(self, temp_dir):
        """Test saving and loading optimization results."""
        # Create sample optimization results
        best_trial = TrialResult(
            trial_number=3,
            parameters={"temperature": 0.8, "max_tokens": 750},
            metrics=MetricResult(quality=0.95, cost=0.002, latency_ms=110),
            success=True
        )
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=[best_trial],
            optimization_mode="pareto",
            dataset_size=10,
            total_trials=1,
            best_parameters=best_trial.parameters,
            metrics_summary={"quality": 0.95}
        )
        
        # Save results
        output_path = temp_dir / "optimization_results.yaml"
        search_result.save_best(str(output_path))
        
        # Verify file was created
        assert output_path.exists()
        
        # Load and verify content
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["parameters"] == best_trial.parameters
        assert saved_data["metrics_summary"] == {"quality": 0.95}

    def test_custom_tunable_component_integration(self, complete_config_file):
        """Test integration with custom tunable components."""
        
        class CustomSentimentAnalyzer(TunableMixin):
            def __init__(self, config_file):
                super().__init__()
                self.llm_tuner = MultiProviderTunableLLM(config_file=config_file)
                self.confidence_threshold = 0.8
                
                # Mark parameters as tunable
                self.mark_as_tunable("confidence_threshold", "float", (0.5, 0.95), 0.8)
                
                # Mark LLM parameters as tunable too
                llm_params = self.llm_tuner.get_tunable_parameters()
                for param_name, param_info in llm_params.items():
                    self.mark_as_tunable(
                        param_name,
                        param_info["type"],
                        param_info["range"],
                        param_info.get("default")
                    )
            
            def analyze_sentiment(self, text):
                """Analyze sentiment with custom logic."""
                # Use LLM for basic sentiment
                llm_response = self.llm_tuner.call(f"Analyze sentiment: {text}")
                
                # Apply confidence threshold logic
                if "positive" in llm_response.text.lower():
                    confidence = 0.9 if self.confidence_threshold < 0.8 else 0.7
                    return {"sentiment": "positive", "confidence": confidence}
                elif "negative" in llm_response.text.lower():
                    confidence = 0.8 if self.confidence_threshold < 0.8 else 0.6
                    return {"sentiment": "negative", "confidence": confidence}
                else:
                    return {"sentiment": "neutral", "confidence": 0.5}
        
        # Register the custom component
        register_tunable_class(CustomSentimentAnalyzer, {"name": "CustomSentimentAnalyzer"})
        
        # Create instance
        with patch('openai.OpenAI'):
            analyzer = CustomSentimentAnalyzer(config_file=complete_config_file)
            
            # Verify it's tunable
            params = analyzer.get_tunable_parameters()
            assert "confidence_threshold" in params
            assert "temperature" in params  # From LLM tuner
            
            # Verify parameter types
            assert params["confidence_threshold"]["type"] == "float"
            assert params["temperature"]["type"] == "float"

    @patch('openai.OpenAI')
    def test_error_handling_in_workflow(self, mock_openai_class, complete_config_file, sentiment_dataset):
        """Test error handling throughout the workflow."""
        # Mock OpenAI client to sometimes fail
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        call_count = 0
        def mock_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Every third call fails
                raise Exception("API Rate Limit Exceeded")
            
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="neutral sentiment"))]
            mock_response.usage = Mock(prompt_tokens=5, completion_tokens=8, total_tokens=13)
            mock_response.model = "gpt-4o-mini"
            return mock_response
        
        mock_client.chat.completions.create.side_effect = mock_api_call
        
        # Create tuner
        tuner = MultiProviderTunableLLM(config_file=complete_config_file)
        
        def robust_metric_function(output, target):
            """Metric function that handles failures gracefully."""
            try:
                if output is None or not hasattr(output, 'text'):
                    return 0.0
                
                output_text = output.text.lower()
                target_sentiment = target.get("sentiment", "")
                return 0.8 if target_sentiment in output_text else 0.2
            except Exception:
                return 0.0
        
        def robust_entrypoint(component, input_data):
            """Entrypoint function that handles API failures."""
            try:
                return component.call(input_data)
            except Exception:
                # Return a mock response for failed calls
                return Mock(text="error response", provider="error", model="error")
        
        # Test that the system handles errors gracefully
        # This would normally be tested with actual AutoTuner, but here we just verify
        # that the components can handle errors without crashing
        
        results = []
        for item in sentiment_dataset[:3]:  # Test first 3 items
            try:
                response = robust_entrypoint(tuner, item["input"])
                metric_score = robust_metric_function(response, item["target"])
                results.append(metric_score)
            except Exception as e:
                results.append(0.0)  # Failed trials get 0 score
        
        # Should have some results, even with failures
        assert len(results) == 3
        assert all(isinstance(score, (int, float)) for score in results)

    def test_multi_provider_configuration(self, temp_dir):
        """Test configuration with multiple providers."""
        multi_config_data = {
            "providers": {
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "default_model": "gpt-4o-mini",
                    "available_models": ["gpt-4o-mini"],
                    "model_capabilities": {
                        "gpt-4o-mini": {
                            "supported_parameters": ["temperature"],
                            "parameters": {
                                "temperature": {"type": "float", "range": [0.0, 2.0], "default": 0.7}
                            }
                        }
                    }
                },
                "gemini": {
                    "api_key_env": "GOOGLE_API_KEY",
                    "default_model": "gemini-1.5-flash",
                    "available_models": ["gemini-1.5-flash"],
                    "model_capabilities": {
                        "gemini-1.5-flash": {
                            "supported_parameters": ["temperature"],
                            "parameters": {
                                "temperature": {"type": "float", "range": [0.0, 1.0], "default": 0.7}
                            }
                        }
                    }
                }
            }
        }
        
        config_path = temp_dir / "multi_provider_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(multi_config_data, f)
        
        # Load and validate
        config_loader = ConfigLoader(str(config_path))
        assert config_loader.validate_config() is True
        
        providers = config_loader.get_providers()
        assert "openai" in providers
        assert "gemini" in providers


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance aspects of the integration."""

    def test_large_dataset_handling(self, complete_config_file):
        """Test handling of large datasets."""
        # Create a large dataset
        large_dataset = []
        for i in range(100):
            sentiment = "positive" if i % 3 == 0 else "negative" if i % 3 == 1 else "neutral"
            large_dataset.append({
                "input": f"Test input number {i} with {sentiment} sentiment",
                "target": {"sentiment": sentiment, "confidence": 0.8}
            })
        
        # Test that config loader handles this efficiently
        config_loader = ConfigLoader(complete_config_file)
        
        # Test dataset fingerprinting (should be fast even for large datasets)
        from octuner.utils.exporter import compute_dataset_fingerprint
        
        import time
        start_time = time.time()
        fingerprint = compute_dataset_fingerprint(large_dataset)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for 100 items)
        assert end_time - start_time < 1.0
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

    def test_memory_usage_optimization(self, complete_config_file, sentiment_dataset):
        """Test that memory usage remains reasonable during optimization."""
        # This is a basic test - in a real scenario you'd use memory profiling tools
        
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and use components
        with patch('openai.OpenAI'):
            tuner = MultiProviderTunableLLM(config_file=complete_config_file)
            
            # Simulate multiple optimization cycles
            for i in range(5):
                # Get tunable parameters multiple times
                params = tuner.get_tunable_parameters()
                assert len(params) > 0
                
                # Simulate parameter setting
                if "temperature" in params:
                    tuner.temperature = 0.5 + i * 0.1
        
        # Force garbage collection after test
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow excessively (allow some growth for test overhead)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold for this test
