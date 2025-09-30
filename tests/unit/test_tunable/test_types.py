"""
Unit tests for tunable type definitions.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from octuner.tunable.types import (
    MetricResult, TrialResult, SearchResult, ScalarizationWeights
)


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test creating MetricResult with required fields."""
        metric = MetricResult(quality=0.85)
        assert metric.quality == 0.85
        assert metric.cost is None
        assert metric.latency_ms is None

    def test_metric_result_with_all_fields(self):
        """Test creating MetricResult with all fields."""
        metric = MetricResult(
            quality=0.9,
            cost=0.005,
            latency_ms=150.0
        )
        assert metric.quality == 0.9
        assert metric.cost == 0.005
        assert metric.latency_ms == 150.0

    def test_metric_result_quality_range(self):
        """Test MetricResult with quality values at boundaries."""
        # Test minimum quality
        metric_min = MetricResult(quality=0.0)
        assert metric_min.quality == 0.0

        # Test maximum quality
        metric_max = MetricResult(quality=1.0)
        assert metric_max.quality == 1.0

    def test_metric_result_negative_values(self):
        """Test MetricResult accepts negative cost/latency (edge case)."""
        metric = MetricResult(
            quality=0.5,
            cost=-0.001,  # Could happen with refunds/credits
            latency_ms=-10.0  # Edge case, but shouldn't break
        )
        assert metric.cost == -0.001
        assert metric.latency_ms == -10.0


class TestTrialResult:
    """Test TrialResult dataclass."""

    def test_trial_result_successful(self):
        """Test creating successful TrialResult."""
        metric = MetricResult(quality=0.85, cost=0.002, latency_ms=120)
        trial = TrialResult(
            trial_number=1,
            parameters={"temperature": 0.7, "top_p": 0.9},
            metrics=metric
        )
        
        assert trial.trial_number == 1
        assert trial.parameters == {"temperature": 0.7, "top_p": 0.9}
        assert trial.metrics == metric
        assert trial.success is True
        assert trial.error is None

    def test_trial_result_failed(self):
        """Test creating failed TrialResult."""
        metric = MetricResult(quality=0.0)
        trial = TrialResult(
            trial_number=5,
            parameters={"temperature": 2.0, "top_p": 0.1},
            metrics=metric,
            success=False,
            error="Token limit exceeded"
        )
        
        assert trial.trial_number == 5
        assert trial.success is False
        assert trial.error == "Token limit exceeded"

    def test_trial_result_empty_parameters(self):
        """Test TrialResult with empty parameters."""
        metric = MetricResult(quality=0.5)
        trial = TrialResult(
            trial_number=0,
            parameters={},
            metrics=metric
        )
        
        assert trial.parameters == {}
        assert trial.success is True


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating SearchResult with all required fields."""
        best_metric = MetricResult(quality=0.9, cost=0.001, latency_ms=100)
        best_trial = TrialResult(
            trial_number=3,
            parameters={"temperature": 0.8, "top_p": 0.95},
            metrics=best_metric
        )
        
        all_trials = [
            TrialResult(1, {"temperature": 0.5}, MetricResult(quality=0.7)),
            TrialResult(2, {"temperature": 0.7}, MetricResult(quality=0.8)),
            best_trial
        ]
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=all_trials,
            optimization_mode="pareto",
            dataset_size=100,
            total_trials=3,
            best_parameters={"temperature": 0.8, "top_p": 0.95},
            metrics_summary={"avg_quality": 0.8, "best_quality": 0.9}
        )
        
        assert search_result.best_trial == best_trial
        assert len(search_result.all_trials) == 3
        assert search_result.optimization_mode == "pareto"
        assert search_result.dataset_size == 100
        assert search_result.total_trials == 3
        assert search_result.best_parameters == {"temperature": 0.8, "top_p": 0.95}
        assert search_result.metrics_summary["avg_quality"] == 0.8

    @patch('octuner.utils.exporter.save_parameters_to_yaml')
    def test_save_best_method(self, mock_save_yaml):
        """Test SearchResult.save_best method."""
        best_metric = MetricResult(quality=0.9)
        best_trial = TrialResult(
            trial_number=1,
            parameters={"temperature": 0.8},
            metrics=best_metric
        )
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=[best_trial],
            optimization_mode="scalarized",
            dataset_size=50,
            total_trials=1,
            best_parameters={"temperature": 0.8},
            metrics_summary={"quality": 0.9}
        )
        
        # Test saving to file
        search_result.save_best("test_params.yaml")
        
        mock_save_yaml.assert_called_once_with(
            {"temperature": 0.8},
            "test_params.yaml",
            {"quality": 0.9}
        )

    def test_search_result_empty_trials(self):
        """Test SearchResult with empty trials list."""
        best_metric = MetricResult(quality=0.0)
        best_trial = TrialResult(
            trial_number=0,
            parameters={},
            metrics=best_metric,
            success=False,
            error="No valid trials"
        )
        
        search_result = SearchResult(
            best_trial=best_trial,
            all_trials=[],
            optimization_mode="constrained",
            dataset_size=0,
            total_trials=0,
            best_parameters={},
            metrics_summary={}
        )
        
        assert len(search_result.all_trials) == 0
        assert search_result.total_trials == 0
        assert search_result.best_parameters == {}


class TestScalarizationWeights:
    """Test ScalarizationWeights dataclass."""

    def test_scalarization_weights_defaults(self):
        """Test ScalarizationWeights with default values."""
        weights = ScalarizationWeights()
        assert weights.cost_weight == 1.0
        assert weights.latency_weight == 1.0

    def test_scalarization_weights_custom(self):
        """Test ScalarizationWeights with custom values."""
        weights = ScalarizationWeights(
            cost_weight=2.5,
            latency_weight=0.5
        )
        assert weights.cost_weight == 2.5
        assert weights.latency_weight == 0.5

    def test_scalarization_weights_zero(self):
        """Test ScalarizationWeights with zero weights."""
        weights = ScalarizationWeights(
            cost_weight=0.0,
            latency_weight=0.0
        )
        assert weights.cost_weight == 0.0
        assert weights.latency_weight == 0.0

    def test_scalarization_weights_negative(self):
        """Test ScalarizationWeights with negative weights (edge case)."""
        weights = ScalarizationWeights(
            cost_weight=-1.0,
            latency_weight=-0.5
        )
        assert weights.cost_weight == -1.0
        assert weights.latency_weight == -0.5


class TestTypeDefinitions:
    """Test type definitions and aliases."""

    def test_param_type_literal(self):
        """Test ParamType literal values."""
        from octuner.tunable.types import ParamType
        
        # These should be valid ParamType values
        valid_types = ["float", "int", "choice", "bool"]
        
        # Note: We can't directly test Literal types at runtime,
        # but we can test that the import works and the type is defined
        assert ParamType is not None

    def test_optimization_mode_literal(self):
        """Test OptimizationMode literal values."""
        from octuner.tunable.types import OptimizationMode
        
        # These should be valid OptimizationMode values
        valid_modes = ["pareto", "constrained", "scalarized"]
        
        # Note: We can't directly test Literal types at runtime,
        # but we can test that the import works and the type is defined
        assert OptimizationMode is not None

    def test_dataset_type_aliases(self):
        """Test dataset type aliases."""
        from octuner.tunable.types import DatasetItem, Dataset
        
        # Test that we can create instances matching the type aliases
        dataset_item = {"input": "test", "target": "output"}
        dataset = [dataset_item, {"input": "test2", "target": "output2"}]
        
        # These should work without type errors
        assert isinstance(dataset_item, dict)
        assert isinstance(dataset, list)
        assert len(dataset) == 2

    def test_function_type_aliases(self):
        """Test function type aliases."""
        from octuner.tunable.types import MetricFunction, EntrypointFunction
        
        # Test that callables match the type aliases
        def sample_metric_function(output, target):
            return 0.5
        
        def sample_entrypoint_function(component, input_data):
            return "result"
        
        # These should be callable
        assert callable(sample_metric_function)
        assert callable(sample_entrypoint_function)
        
        # Test function calls
        metric_result = sample_metric_function("output", "target")
        assert metric_result == 0.5
        
        entrypoint_result = sample_entrypoint_function("component", "input")
        assert entrypoint_result == "result"

    def test_constraints_type_alias(self):
        """Test Constraints type alias."""
        from octuner.tunable.types import Constraints
        
        # Test that dictionary matches Constraints type
        constraints = {
            "latency_ms": 1000,
            "cost_total": 0.01,
            "memory_mb": 512
        }
        
        assert isinstance(constraints, dict)
        assert constraints["latency_ms"] == 1000
        assert constraints["cost_total"] == 0.01
