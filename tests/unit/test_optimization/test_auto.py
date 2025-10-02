"""
Unit tests for AutoTuner functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from octuner.optimization.auto import AutoTuner
from octuner.tunable.types import SearchResult, TrialResult, MetricResult, OptimizationMode
from octuner.tunable.mixin import TunableMixin


class TestAutoTuner:
    """Test AutoTuner class functionality."""

    @pytest.fixture
    def mock_tunable_component(self):
        """Create a mock tunable component."""
        component = Mock()
        # Add the attributes that the component needs
        component.temperature = 0.7
        component.max_tokens = 1000
        component.top_p = 1.0
        # Add get_tunable_parameters method that returns some tunable parameters
        component.get_tunable_parameters = Mock(return_value={
            "temperature": {"type": "float", "range": (0.0, 2.0), "default": 0.7},
            "max_tokens": {"type": "int", "range": (50, 4000), "default": 1000},
            "top_p": {"type": "float", "range": (0.0, 1.0), "default": 1.0}
        })
        return component

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for optimization."""
        return [
            {"input": "I love this product!", "target": {"sentiment": "positive", "confidence": 0.9}},
            {"input": "This is terrible", "target": {"sentiment": "negative", "confidence": 0.8}},
            {"input": "It's okay", "target": {"sentiment": "neutral", "confidence": 0.6}},
            {"input": "Amazing quality!", "target": {"sentiment": "positive", "confidence": 0.95}},
            {"input": "Worst purchase ever", "target": {"sentiment": "negative", "confidence": 0.9}}
        ]

    @pytest.fixture
    def sample_metric_function(self):
        """Sample metric function for optimization."""
        def metric_func(output, target):
            # Simple sentiment accuracy metric
            if hasattr(output, 'text'):
                output_text = output.text.lower()
            else:
                output_text = str(output).lower()
            
            target_sentiment = target.get("sentiment", "")
            
            if target_sentiment in output_text:
                return 0.9
            else:
                return 0.1
        
        return metric_func

    @pytest.fixture
    def sample_entrypoint_function(self):
        """Sample entrypoint function for optimization."""
        def entrypoint_func(component, input_data):
            return component.call(input_data)
        
        return entrypoint_func

    @patch('octuner.tunable.mixin.get_tunable_parameters')
    def test_autotuner_init_basic(self, mock_get_tunable_parameters, mock_tunable_component, sample_dataset, 
                                 sample_metric_function, sample_entrypoint_function):
        """Test AutoTuner initialization with basic parameters."""
        # Mock the get_tunable_parameters function
        mock_get_tunable_parameters.return_value = {
            "temperature": ("float", 0.0, 2.0),
            "max_tokens": ("int", 50, 4000),
            "top_p": ("float", 0.0, 1.0)
        }
        
        tuner = AutoTuner(
            component=mock_tunable_component,
            entrypoint=sample_entrypoint_function,
            dataset=sample_dataset,
            metric=sample_metric_function
        )
        
        assert tuner.component == mock_tunable_component
        assert tuner.dataset == sample_dataset
        assert tuner.metric == sample_metric_function
        assert tuner.entrypoint == sample_entrypoint_function
        assert tuner.max_workers == 1  # default

    def test_autotuner_init_with_parameters(self, mock_tunable_component, sample_dataset,
                                          sample_metric_function, sample_entrypoint_function):
        """Test AutoTuner initialization with custom parameters."""
        constraints = {"latency_ms": 500, "cost_total": 0.01}
        
        tuner = AutoTuner(
            component=mock_tunable_component,
            entrypoint=sample_entrypoint_function,
            dataset=sample_dataset,
            metric=sample_metric_function
        )
        
        # Test that the tuner was created successfully
        assert tuner.component == mock_tunable_component
        assert tuner.dataset == sample_dataset
        assert tuner.metric == sample_metric_function
        assert tuner.entrypoint == sample_entrypoint_function

    def test_autotuner_init_non_tunable_component(self, sample_dataset, sample_metric_function, sample_entrypoint_function):
        """Test AutoTuner initialization with non-tunable component."""
        non_tunable_component = Mock()
        non_tunable_component.get_tunable_parameters.return_value = {}
        
        with pytest.raises(ValueError) as exc_info:
            AutoTuner(
                component=non_tunable_component,
                dataset=sample_dataset,
                metric_function=sample_metric_function,
                entrypoint_function=sample_entrypoint_function
            )
        
        assert "no tunable parameters" in str(exc_info.value).lower()

    def test_autotuner_init_empty_dataset(self, mock_tunable_component, sample_metric_function, sample_entrypoint_function):
        """Test AutoTuner initialization with empty dataset."""
        with pytest.raises(ValueError) as exc_info:
            AutoTuner(
                component=mock_tunable_component,
                dataset=[],
                metric_function=sample_metric_function,
                entrypoint_function=sample_entrypoint_function
            )
        
        assert "empty" in str(exc_info.value).lower()

    @patch('octuner.discovery.discovery.get_tunable_parameters')
    @patch('octuner.optimization.auto.LLMOptimizer')
    def test_optimize_basic(self, mock_optimizer_class, mock_get_tunable_parameters, mock_tunable_component, sample_dataset,
                           sample_metric_function, sample_entrypoint_function):
        """Test basic optimization process."""
        # Mock get_tunable_parameters
        mock_get_tunable_parameters.return_value = {
            "temperature": ("float", 0.0, 2.0),
            "max_tokens": ("int", 50, 4000),
            "top_p": ("float", 0.0, 1.0)
        }
        
        # Mock optimization result
        best_trial = TrialResult(
            trial_number=5,
            parameters={"temperature": 0.8, "max_tokens": 500, "top_p": 0.9},
            metrics=MetricResult(quality=0.95, cost=0.002, latency_ms=120),
            success=True
        )
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.get_best_trial.return_value = best_trial
        mock_optimizer.get_best_parameters.return_value = best_trial.parameters
        
        all_trials = [
            TrialResult(1, {"temperature": 0.5}, MetricResult(quality=0.7), True),
            TrialResult(2, {"temperature": 0.7}, MetricResult(quality=0.8), True),
            best_trial
        ]
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=all_trials,
            optimization_mode="pareto",
            dataset_size=5,
            total_trials=3,
            best_parameters=best_trial.parameters,
            metrics_summary={"avg_quality": 0.82, "best_quality": 0.95}
        )
        
        mock_optimizer.optimize.return_value = search_result
        
        tuner = AutoTuner(
            component=mock_tunable_component,
            dataset=sample_dataset,
            metric=sample_metric_function,
            entrypoint=sample_entrypoint_function
        )
        
        result = tuner.search()
        
        assert isinstance(result, SearchResult)
        assert result.best_trial == best_trial
        assert len(result.all_trials) == 3
        assert result.dataset_size == 5
        
        # Verify optimizer was called correctly
        mock_optimizer.optimize.assert_called_once()

    @patch('octuner.discovery.discovery.get_tunable_parameters')
    @patch('octuner.optimization.auto.LLMOptimizer')
    def test_optimize_with_constraints(self, mock_optimizer_class, mock_get_tunable_parameters, mock_tunable_component, sample_dataset,
                                     sample_metric_function, sample_entrypoint_function):
        """Test optimization with constraints."""
        # Mock get_tunable_parameters
        mock_get_tunable_parameters.return_value = {
            "temperature": ("float", 0.0, 2.0),
            "max_tokens": ("int", 50, 4000),
            "top_p": ("float", 0.0, 1.0)
        }
        
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock constrained optimization result
        best_trial = TrialResult(
            trial_number=3,
            parameters={"temperature": 0.6, "max_tokens": 200},
            metrics=MetricResult(quality=0.85, cost=0.001, latency_ms=80),
            success=True
        )
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=[best_trial],
            optimization_mode="constrained",
            dataset_size=5,
            total_trials=1,
            best_parameters=best_trial.parameters,
            metrics_summary={"quality": 0.85}
        )
        
        mock_optimizer.optimize.return_value = search_result
        
        constraints = {"latency_ms": 100, "cost_total": 0.005}
        
        tuner = AutoTuner(
            component=mock_tunable_component,
            dataset=sample_dataset,
            metric=sample_metric_function,
            entrypoint=sample_entrypoint_function,
            optimization_mode="constrained",
            constraints=constraints
        )
        
        result = tuner.search(mode="constrained", constraints=constraints)
        
        assert isinstance(result, SearchResult)
        assert result.optimization_mode == "constrained"
        
        # Verify optimizer was initialized with constraints
        mock_optimizer_class.assert_called_once()
        init_args = mock_optimizer_class.call_args
        assert "constraints" in init_args[1] or any("constraints" in str(arg) for arg in init_args[0])

    @patch('octuner.discovery.discovery.get_tunable_parameters')
    @patch('octuner.optimization.auto.LLMOptimizer')
    def test_optimize_with_scalarization(self, mock_optimizer_class, mock_get_tunable_parameters, mock_tunable_component, sample_dataset,
                                       sample_metric_function, sample_entrypoint_function):
        """Test optimization with scalarization weights."""
        # Mock get_tunable_parameters
        mock_get_tunable_parameters.return_value = {
            "temperature": ("float", 0.0, 2.0),
            "max_tokens": ("int", 50, 4000),
            "top_p": ("float", 0.0, 1.0)
        }
        
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        from octuner.tunable.types import ScalarizationWeights
        weights = ScalarizationWeights(cost_weight=2.0, latency_weight=1.0)
        
        best_trial = TrialResult(
            trial_number=2,
            parameters={"temperature": 0.5},
            metrics=MetricResult(quality=0.8, cost=0.001, latency_ms=90),
            success=True
        )
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=[best_trial],
            optimization_mode="scalarized",
            dataset_size=5,
            total_trials=1,
            best_parameters=best_trial.parameters,
            metrics_summary={"scalarized_score": 0.75}
        )
        
        mock_optimizer.optimize.return_value = search_result
        
        tuner = AutoTuner(
            component=mock_tunable_component,
            dataset=sample_dataset,
            metric=sample_metric_function,
            entrypoint=sample_entrypoint_function,
            optimization_mode="scalarized",
            scalarization_weights=weights
        )
        
        result = tuner.search(mode="scalarized", scalarization_weights=weights)
        
        assert isinstance(result, SearchResult)
        assert result.optimization_mode == "scalarized"

    @patch('octuner.discovery.discovery.get_tunable_parameters')
    @patch('octuner.optimization.auto.LLMOptimizer')
    def test_optimize_with_failed_trials(self, mock_optimizer_class, mock_get_tunable_parameters, mock_tunable_component, sample_dataset,
                                       sample_metric_function, sample_entrypoint_function):
        """Test optimization with some failed trials."""
        # Mock get_tunable_parameters
        mock_get_tunable_parameters.return_value = {
            "temperature": ("float", 0.0, 2.0),
            "max_tokens": ("int", 50, 4000),
            "top_p": ("float", 0.0, 1.0)
        }
        
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Include failed trials in results
        failed_trial = TrialResult(
            trial_number=2,
            parameters={"temperature": 3.0},  # Invalid temperature
            metrics=MetricResult(quality=0.0),
            success=False,
            error="Invalid parameter value"
        )
        
        successful_trial = TrialResult(
            trial_number=3,
            parameters={"temperature": 0.8},
            metrics=MetricResult(quality=0.9, cost=0.002, latency_ms=110),
            success=True
        )
        
        # Set up mock optimizer return values
        mock_optimizer.get_best_trial.return_value = successful_trial
        mock_optimizer.get_best_parameters.return_value = successful_trial.parameters
        
        all_trials = [failed_trial, successful_trial]
        
        search_result = SearchResult(
            best_trial=successful_trial,
            all_trials=all_trials,
            optimization_mode="pareto",
            dataset_size=5,
            total_trials=2,
            best_parameters=successful_trial.parameters,
            metrics_summary={"success_rate": 0.5, "best_quality": 0.9}
        )
        
        mock_optimizer.optimize.return_value = search_result
        
        tuner = AutoTuner(
            component=mock_tunable_component,
            dataset=sample_dataset,
            metric=sample_metric_function,
            entrypoint=sample_entrypoint_function
        )
        
        result = tuner.search()
        
        assert isinstance(result, SearchResult)
        assert len(result.all_trials) == 2
        assert result.best_trial.success is True
        assert not all(trial.success for trial in result.all_trials)

    def test_get_parameter_suggestions(self, mock_tunable_component, sample_dataset,
                                     sample_metric_function, sample_entrypoint_function):
        """Test getting parameter suggestions based on tunable parameters."""
        tuner = AutoTuner(
            component=mock_tunable_component,
            dataset=sample_dataset,
            metric=sample_metric_function,
            entrypoint=sample_entrypoint_function
        )
        
        suggestions = tuner._get_parameter_suggestions()
        
        assert isinstance(suggestions, dict)
        assert "temperature" in suggestions
        assert "max_tokens" in suggestions
        assert "top_p" in suggestions
        
        # Check parameter types and ranges
        temp_suggestion = suggestions["temperature"]
        assert temp_suggestion["type"] == "float"
        assert temp_suggestion["range"] == (0.0, 2.0)

    def test_validate_optimization_mode(self, mock_tunable_component, sample_dataset,
                                      sample_metric_function, sample_entrypoint_function):
        """Test validation of optimization mode."""
        # Valid modes should work
        valid_modes = ["pareto", "constrained", "scalarized"]
        
        for mode in valid_modes:
            tuner = AutoTuner(
                component=mock_tunable_component,
                dataset=sample_dataset,
                metric_function=sample_metric_function,
                entrypoint_function=sample_entrypoint_function,
                optimization_mode=mode
            )
            assert tuner.optimization_mode == mode
        
        # Invalid mode should raise error
        with pytest.raises(ValueError) as exc_info:
            AutoTuner(
                component=mock_tunable_component,
                dataset=sample_dataset,
                metric_function=sample_metric_function,
                entrypoint_function=sample_entrypoint_function,
                optimization_mode="invalid_mode"
            )
        
        assert "invalid optimization mode" in str(exc_info.value).lower()

    def test_validate_n_trials(self, mock_tunable_component, sample_dataset,
                              sample_metric_function, sample_entrypoint_function):
        """Test validation of n_trials parameter."""
        # Positive values should work
        tuner = AutoTuner(
            component=mock_tunable_component,
            dataset=sample_dataset,
            metric=sample_metric_function,
            entrypoint=sample_entrypoint_function,
            n_trials=50
        )
        assert tuner.n_trials == 50
        
        # Zero or negative values should raise error
        with pytest.raises(ValueError) as exc_info:
            AutoTuner(
                component=mock_tunable_component,
                dataset=sample_dataset,
                metric_function=sample_metric_function,
                entrypoint_function=sample_entrypoint_function,
                n_trials=0
            )
        
        assert "n_trials must be positive" in str(exc_info.value).lower()

    @patch('octuner.optimization.auto.LLMOptimizer')
    def test_optimization_progress_tracking(self, mock_optimizer_class, mock_tunable_component, 
                                          sample_dataset, sample_metric_function, sample_entrypoint_function):
        """Test that optimization progress can be tracked."""
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock progressive results
        trial_results = []
        for i in range(5):
            trial_results.append(TrialResult(
                trial_number=i+1,
                parameters={"temperature": 0.5 + i*0.1},
                metrics=MetricResult(quality=0.7 + i*0.05),
                success=True
            ))
        
        search_result = SearchResult(
            best_trial=trial_results[-1],  # Last trial is best
            all_trials=trial_results,
            optimization_mode="pareto",
            dataset_size=5,
            total_trials=5,
            best_parameters=trial_results[-1].parameters,
            metrics_summary={"quality_improvement": 0.2}
        )
        
        mock_optimizer.optimize.return_value = search_result
        
        tuner = AutoTuner(
            component=mock_tunable_component,
            dataset=sample_dataset,
            metric=sample_metric_function,
            entrypoint=sample_entrypoint_function,
            n_trials=5
        )
        
        result = tuner.search()
        
        assert len(result.all_trials) == 5
        assert result.total_trials == 5
        
        # Verify quality improvement over trials
        qualities = [trial.metrics.quality for trial in result.all_trials if trial.success]
        assert len(qualities) == 5
        assert qualities[-1] > qualities[0]  # Should improve over time


class TestAutoTunerEdgeCases:
    """Test edge cases and error conditions for AutoTuner."""

    def test_autotuner_with_single_parameter(self):
        """Test AutoTuner with component that has only one tunable parameter."""
        component = Mock(spec=TunableMixin)
        component.get_tunable_parameters.return_value = {
            "temperature": {"type": "float", "range": (0.0, 2.0), "default": 0.7}
        }
        component.is_tunable.return_value = True
        
        dataset = [{"input": "test", "target": {"expected": "output"}}]
        
        def metric_func(output, target):
            return 0.8
        
        def entrypoint_func(component, input_data):
            return component.call(input_data)
        
        tuner = AutoTuner(
            component=component,
            dataset=dataset,
            metric_function=metric_func,
            entrypoint_function=entrypoint_func
        )
        
        suggestions = tuner._get_parameter_suggestions()
        assert len(suggestions) == 1
        assert "temperature" in suggestions

    def test_autotuner_with_choice_parameters(self):
        """Test AutoTuner with choice-type parameters."""
        component = Mock(spec=TunableMixin)
        component.get_tunable_parameters.return_value = {
            "model": {"type": "choice", "range": ("gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"), "default": "gpt-4o-mini"},
            "use_websearch": {"type": "bool", "range": (True, False), "default": False}
        }
        component.is_tunable.return_value = True
        
        dataset = [{"input": "test", "target": {"expected": "output"}}]
        
        def metric_func(output, target):
            return 0.8
        
        def entrypoint_func(component, input_data):
            return component.call(input_data)
        
        tuner = AutoTuner(
            component=component,
            dataset=dataset,
            metric_function=metric_func,
            entrypoint_function=entrypoint_func
        )
        
        suggestions = tuner._get_parameter_suggestions()
        assert "model" in suggestions
        assert "use_websearch" in suggestions
        assert suggestions["model"]["type"] == "choice"
        assert suggestions["use_websearch"]["type"] == "bool"

    def test_autotuner_with_large_dataset(self):
        """Test AutoTuner behavior with large dataset."""
        component = Mock(spec=TunableMixin)
        component.get_tunable_parameters.return_value = {
            "temperature": {"type": "float", "range": (0.0, 2.0), "default": 0.7}
        }
        component.is_tunable.return_value = True
        
        # Create large dataset
        large_dataset = [
            {"input": f"test input {i}", "target": {"expected": f"output {i}"}}
            for i in range(1000)
        ]
        
        def metric_func(output, target):
            return 0.8
        
        def entrypoint_func(component, input_data):
            return component.call(input_data)
        
        tuner = AutoTuner(
            component=component,
            dataset=large_dataset,
            metric_function=metric_func,
            entrypoint_function=entrypoint_func
        )
        
        assert tuner.dataset == large_dataset
        assert len(tuner.dataset) == 1000
