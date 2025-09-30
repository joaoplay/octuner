"""
Unit tests for exporter utility functions.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from octuner.utils.exporter import (
    save_parameters_to_yaml, apply_best, create_metadata_summary, compute_dataset_fingerprint
)
from octuner.tunable.types import SearchResult, TrialResult, MetricResult


class TestSaveParametersToYaml:
    """Test save_parameters_to_yaml function."""

    def test_save_parameters_basic(self, temp_dir):
        """Test basic parameter saving to YAML."""
        parameters = {
            "temperature": 0.8,
            "max_tokens": 500,
            "top_p": 0.9
        }
        
        metrics_summary = {
            "quality": 0.95,
            "cost": 0.002,
            "latency_ms": 120
        }
        
        output_path = temp_dir / "test_params.yaml"
        
        save_parameters_to_yaml(parameters, str(output_path), metrics_summary)
        
        # Verify file was created and contains expected content
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["parameters"] == parameters
        assert saved_data["metrics_summary"] == metrics_summary
        assert "metadata" in saved_data
        assert "timestamp" in saved_data["metadata"]

    def test_save_parameters_without_metrics(self, temp_dir):
        """Test saving parameters without metrics summary."""
        parameters = {"temperature": 0.7}
        output_path = temp_dir / "params_no_metrics.yaml"
        
        save_parameters_to_yaml(parameters, str(output_path))
        
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["parameters"] == parameters
        assert saved_data["metrics_summary"] is None

    def test_save_parameters_empty_dict(self, temp_dir):
        """Test saving empty parameters dictionary."""
        parameters = {}
        output_path = temp_dir / "empty_params.yaml"
        
        save_parameters_to_yaml(parameters, str(output_path))
        
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["parameters"] == {}

    def test_save_parameters_complex_types(self, temp_dir):
        """Test saving parameters with complex types."""
        parameters = {
            "model": "gpt-4o-mini",
            "temperature": 0.8,
            "use_websearch": True,
            "models_list": ["gpt-3.5-turbo", "gpt-4o"],
            "nested_config": {"option1": "value1", "option2": 42}
        }
        
        output_path = temp_dir / "complex_params.yaml"
        
        save_parameters_to_yaml(parameters, str(output_path))
        
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["parameters"] == parameters
        assert saved_data["parameters"]["use_websearch"] is True
        assert saved_data["parameters"]["models_list"] == ["gpt-3.5-turbo", "gpt-4o"]

    def test_save_parameters_overwrite_existing(self, temp_dir):
        """Test overwriting existing parameter file."""
        output_path = temp_dir / "overwrite_test.yaml"
        
        # Save first set of parameters
        params1 = {"temperature": 0.5}
        save_parameters_to_yaml(params1, str(output_path))
        
        # Save second set of parameters (should overwrite)
        params2 = {"temperature": 0.9, "max_tokens": 1000}
        save_parameters_to_yaml(params2, str(output_path))
        
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["parameters"] == params2

    def test_save_parameters_invalid_path(self):
        """Test saving to invalid path."""
        parameters = {"temperature": 0.8}
        invalid_path = "/invalid/path/that/does/not/exist/params.yaml"
        
        with pytest.raises((FileNotFoundError, OSError, PermissionError)):
            save_parameters_to_yaml(parameters, invalid_path)


class TestApplyBest:
    """Test apply_best function."""

    def test_apply_best_with_search_result(self):
        """Test applying best parameters from SearchResult."""
        from octuner.tunable.mixin import TunableMixin
        from octuner.tunable.registry import register_tunable_class
        
        class TestComponent(TunableMixin):
            def __init__(self):
                super().__init__()
                self.temperature = 0.5
                self.max_tokens = 100
        
        # Register the component as tunable
        register_tunable_class(
            TestComponent,
            {
                "temperature": ("float", 0.0, 1.0),
                "max_tokens": ("int", 1, 1000)
            }
        )
        
        component = TestComponent()
        
        # Create SearchResult with best parameters
        best_trial = TrialResult(
            trial_number=5,
            parameters={"temperature": 0.8, "max_tokens": 500},
            metrics=MetricResult(quality=0.95),
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
        
        apply_best(component, search_result)
        
        # Verify parameters were set on component
        assert component.temperature == 0.8
        assert component.max_tokens == 500

    def test_apply_best_with_parameter_dict(self):
        """Test applying best parameters from dictionary."""
        from octuner.tunable.mixin import TunableMixin
        from octuner.tunable.registry import register_tunable_class
        
        class TestComponent(TunableMixin):
            def __init__(self):
                super().__init__()
                self.temperature = 0.5
                self.top_p = 0.8
                self.use_websearch = False
        
        # Register the component as tunable
        register_tunable_class(
            TestComponent,
            {
                "temperature": ("float", 0.0, 1.0),
                "top_p": ("float", 0.0, 1.0),
                "use_websearch": ("bool", None, None)
            }
        )
        
        component = TestComponent()
        
        parameters = {
            "temperature": 0.7,
            "top_p": 0.9,
            "use_websearch": True
        }
        
        apply_best(component, parameters)
        
        # Verify parameters were set
        assert component.temperature == 0.7
        assert component.top_p == 0.9
        assert component.use_websearch is True

    def test_apply_best_empty_parameters(self):
        """Test applying empty parameters."""
        from octuner.tunable.mixin import TunableMixin
        
        class TestComponent(TunableMixin):
            def __init__(self):
                super().__init__()
        
        component = TestComponent()
        
        apply_best(component, {})
        
        # Should not set any attributes on component
        # Check that no unexpected setattr calls were made
        # (This is a bit tricky to test directly, but no exception should be raised)

    def test_apply_best_none_values(self):
        """Test applying parameters with None values."""
        from octuner.tunable.mixin import TunableMixin
        from octuner.tunable.registry import register_tunable_class
        
        class TestComponent(TunableMixin):
            def __init__(self):
                super().__init__()
                self.temperature = 0.5
                self.max_tokens = 100
                self.model = "gpt-3.5"
        
        # Register the component as tunable
        register_tunable_class(
            TestComponent,
            {
                "temperature": ("float", 0.0, 1.0),
                "max_tokens": ("int", 1, 1000),
                "model": ("choice", ["gpt-3.5", "gpt-4o"])
            }
        )
        
        component = TestComponent()
        
        parameters = {
            "temperature": 0.8,
            "max_tokens": None,
            "model": "gpt-4o"
        }
        
        apply_best(component, parameters)
        
        # Should set all parameters, including None values
        assert component.temperature == 0.8
        assert component.max_tokens is None
        assert component.model == "gpt-4o"

    def test_apply_best_with_complex_component(self):
        """Test applying parameters to component with existing attributes."""
        from octuner.tunable.mixin import TunableMixin
        from octuner.tunable.registry import register_tunable_class
        
        class MockComponent(TunableMixin):
            def __init__(self):
                super().__init__()
                self.existing_attr = "existing_value"
                self.temperature = 0.5
        
        # Register the component as tunable
        register_tunable_class(
            MockComponent,
            {
                "temperature": ("float", 0.0, 1.0),
                "new_attribute": ("choice", ["new_value", "other_value"])
            }
        )
        
        component = MockComponent()
        
        parameters = {
            "temperature": 0.9,
            "new_attribute": "new_value"
        }
        
        apply_best(component, parameters)
        
        # Should update existing and add new attributes
        assert component.existing_attr == "existing_value"  # unchanged
        assert component.temperature == 0.9  # updated
        assert component.new_attribute == "new_value"  # new

    def test_apply_best_invalid_input_type(self):
        """Test apply_best with invalid input type."""
        component = Mock()
        
        # Should handle invalid input gracefully or raise appropriate error
        with pytest.raises(FileNotFoundError):
            apply_best(component, "invalid_input")


class TestCreateMetadataSummary:
    """Test create_metadata_summary function."""

    def test_create_metadata_summary_basic(self):
        """Test creating metadata summary with basic trial results."""
        trials = [
            TrialResult(1, {"temp": 0.5}, MetricResult(quality=0.8, cost=0.001, latency_ms=100), True),
            TrialResult(2, {"temp": 0.7}, MetricResult(quality=0.85, cost=0.002, latency_ms=120), True),
            TrialResult(3, {"temp": 0.9}, MetricResult(quality=0.75, cost=0.003, latency_ms=140), True)
        ]
        
        summary = create_metadata_summary(trials)
        
        assert isinstance(summary, dict)
        assert summary["total_trials"] == 3
        assert summary["successful_trials"] == 3
        assert summary["success_rate"] == 1.0
        assert summary["avg_quality"] == round((0.8 + 0.85 + 0.75) / 3, 3)
        assert summary["best_quality"] == 0.85
        assert summary["total_cost"] == 0.001 + 0.002 + 0.003
        assert summary["avg_latency_ms"] == (100 + 120 + 140) / 3

    def test_create_metadata_summary_with_failures(self):
        """Test creating metadata summary with failed trials."""
        trials = [
            TrialResult(1, {"temp": 0.5}, MetricResult(quality=0.8), True),
            TrialResult(2, {"temp": 2.5}, MetricResult(quality=0.0), False, "Invalid parameter"),
            TrialResult(3, {"temp": 0.7}, MetricResult(quality=0.9), True)
        ]
        
        summary = create_metadata_summary(trials)
        
        assert summary["total_trials"] == 3
        assert summary["successful_trials"] == 2
        assert summary["success_rate"] == 2/3
        assert summary["avg_quality"] == round((0.8 + 0.9) / 2, 3)  # Only successful trials
        assert summary["best_quality"] == 0.9

    def test_create_metadata_summary_empty_trials(self):
        """Test creating metadata summary with empty trials list."""
        summary = create_metadata_summary([])
        
        assert summary["total_trials"] == 0
        assert summary["successful_trials"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["avg_quality"] == 0.0
        assert summary["best_quality"] == 0.0
        assert summary["total_cost"] == 0.0
        assert summary["avg_latency_ms"] == 0.0

    def test_create_metadata_summary_none_values(self):
        """Test creating metadata summary with None values in metrics."""
        trials = [
            TrialResult(1, {"temp": 0.5}, MetricResult(quality=0.8, cost=None, latency_ms=None), True),
            TrialResult(2, {"temp": 0.7}, MetricResult(quality=0.9, cost=0.002, latency_ms=120), True)
        ]
        
        summary = create_metadata_summary(trials)
        
        assert summary["total_trials"] == 2
        assert summary["avg_quality"] == round((0.8 + 0.9) / 2, 3)
        assert summary["total_cost"] == 0.002  # Only count non-None values
        assert summary["avg_latency_ms"] == 120  # Only count non-None values

    def test_create_metadata_summary_all_failed(self):
        """Test creating metadata summary when all trials failed."""
        trials = [
            TrialResult(1, {"temp": 2.5}, MetricResult(quality=0.0), False, "Error 1"),
            TrialResult(2, {"temp": 3.0}, MetricResult(quality=0.0), False, "Error 2")
        ]
        
        summary = create_metadata_summary(trials)
        
        assert summary["total_trials"] == 2
        assert summary["successful_trials"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["avg_quality"] == 0.0
        assert summary["best_quality"] == 0.0


class TestComputeDatasetFingerprint:
    """Test compute_dataset_fingerprint function."""

    def test_compute_dataset_fingerprint_basic(self):
        """Test computing fingerprint for basic dataset."""
        dataset = [
            {"input": "Hello world", "target": {"label": "greeting"}},
            {"input": "Goodbye", "target": {"label": "farewell"}},
            {"input": "How are you?", "target": {"label": "question"}}
        ]
        
        fingerprint = compute_dataset_fingerprint(dataset)
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0
        
        # Same dataset should produce same fingerprint
        fingerprint2 = compute_dataset_fingerprint(dataset)
        assert fingerprint == fingerprint2

    def test_compute_dataset_fingerprint_different_datasets(self):
        """Test that different datasets produce different fingerprints."""
        dataset1 = [
            {"input": "Hello", "target": {"label": "greeting"}}
        ]
        
        dataset2 = [
            {"input": "Goodbye", "target": {"label": "farewell"}}
        ]
        
        fingerprint1 = compute_dataset_fingerprint(dataset1)
        fingerprint2 = compute_dataset_fingerprint(dataset2)
        
        assert fingerprint1 != fingerprint2

    def test_compute_dataset_fingerprint_order_sensitive(self):
        """Test that dataset order affects fingerprint."""
        dataset1 = [
            {"input": "A", "target": {"label": "1"}},
            {"input": "B", "target": {"label": "2"}}
        ]
        
        dataset2 = [
            {"input": "B", "target": {"label": "2"}},
            {"input": "A", "target": {"label": "1"}}
        ]
        
        fingerprint1 = compute_dataset_fingerprint(dataset1)
        fingerprint2 = compute_dataset_fingerprint(dataset2)
        
        # Order should matter for fingerprint
        assert fingerprint1 != fingerprint2

    def test_compute_dataset_fingerprint_empty_dataset(self):
        """Test computing fingerprint for empty dataset."""
        fingerprint = compute_dataset_fingerprint([])
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

    def test_compute_dataset_fingerprint_complex_data(self):
        """Test computing fingerprint for dataset with complex data types."""
        dataset = [
            {
                "input": {"text": "Hello", "metadata": {"lang": "en"}},
                "target": {
                    "sentiment": "positive",
                    "confidence": 0.9,
                    "categories": ["greeting", "polite"]
                }
            }
        ]
        
        fingerprint = compute_dataset_fingerprint(dataset)
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

    def test_compute_dataset_fingerprint_consistency(self):
        """Test fingerprint consistency across multiple calls."""
        dataset = [
            {"input": "Test input", "target": {"output": "Test output"}}
        ]
        
        fingerprints = []
        for _ in range(5):
            fingerprints.append(compute_dataset_fingerprint(dataset))
        
        # All fingerprints should be identical
        assert all(fp == fingerprints[0] for fp in fingerprints)

    def test_compute_dataset_fingerprint_with_special_chars(self):
        """Test fingerprint with special characters and unicode."""
        dataset = [
            {"input": "Hello 世界! 🌍", "target": {"label": "unicode"}},
            {"input": "Special chars: @#$%^&*()", "target": {"label": "special"}}
        ]
        
        fingerprint = compute_dataset_fingerprint(dataset)
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

    def test_compute_dataset_fingerprint_large_dataset(self):
        """Test fingerprint computation for large dataset."""
        # Create large dataset
        large_dataset = [
            {"input": f"Input {i}", "target": {"output": f"Output {i}"}}
            for i in range(1000)
        ]
        
        fingerprint = compute_dataset_fingerprint(large_dataset)
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0
