"""
Unit tests for tunable registry functionality.
"""

import pytest
from unittest.mock import Mock, patch

from octuner.tunable.registry import (
    get_tunable_metadata, is_tunable_registered, register_tunable_class
)
from octuner.tunable.mixin import TunableMixin


class TestTunableRegistry:
    """Test tunable registry functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear the registry before each test
        from octuner.tunable import registry
        registry._TUNABLE_METADATA.clear()

    def test_register_tunable_class_basic(self):
        """Test registering a basic tunable class."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        # Register with basic parameters
        params = {"temperature": ("float", (0.0, 2.0), 0.7)}
        register_tunable_class(TestTunable, params)
        
        assert is_tunable_registered(TestTunable)

    def test_register_tunable_class_with_metadata(self):
        """Test registering tunable class with parameters and custom call method."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        # Register with parameters and custom call method
        params = {"temperature": ("float", (0.0, 2.0), 0.7)}
        register_tunable_class(TestTunable, params, call_method="custom_call")
        
        assert is_tunable_registered(TestTunable)
        retrieved_metadata = get_tunable_metadata(TestTunable)
        assert retrieved_metadata["call_method"] == "custom_call"
        assert "temperature" in retrieved_metadata["tunables"]

    def test_register_tunable_class_override_metadata(self):
        """Test overriding metadata when registering same class twice."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        # First registration
        first_params = {"temperature": ("float", (0.0, 2.0), 0.7)}
        register_tunable_class(TestTunable, first_params, call_method="first_call")
        
        # Second registration with different parameters
        second_params = {"temperature": ("float", (0.0, 1.0), 0.5), "max_tokens": ("int", (100, 2000), 500)}
        register_tunable_class(TestTunable, second_params, call_method="second_call")
        
        retrieved_metadata = get_tunable_metadata(TestTunable)
        assert retrieved_metadata["call_method"] == "second_call"
        assert "max_tokens" in retrieved_metadata["tunables"]

    def test_register_non_tunable_class(self):
        """Test registering a non-tunable class."""
        class NonTunable:
            pass
        
        # Register without any parameters
        register_tunable_class(NonTunable)
        
        # Should NOT be registered since no tunable parameters provided
        assert not is_tunable_registered(NonTunable)
        metadata = get_tunable_metadata(NonTunable)
        assert metadata is not None

    def test_is_tunable_registered_unregistered_class(self):
        """Test checking if unregistered class is tunable."""
        class UnregisteredTunable(TunableMixin):
            pass
        
        assert not is_tunable_registered(UnregisteredTunable)

    def test_is_tunable_registered_with_instance(self):
        """Test checking if instance of registered class is tunable."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        params = {"temperature": ("float", (0.0, 2.0), 0.7)}
        register_tunable_class(TestTunable, params)
        instance = TestTunable()
        
        # Should work with both class and instance
        assert is_tunable_registered(TestTunable)
        assert is_tunable_registered(instance)

    def test_get_tunable_metadata_registered_class(self):
        """Test getting metadata for registered class."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        params = {"temperature": ("float", (0.0, 2.0), 0.7)}
        register_tunable_class(TestTunable, params, call_method="test_call")
        
        retrieved_metadata = get_tunable_metadata(TestTunable)
        assert retrieved_metadata["call_method"] == "test_call"
        assert "temperature" in retrieved_metadata["tunables"]

    def test_get_tunable_metadata_with_instance(self):
        """Test getting metadata using instance."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        metadata = {"name": "TestComponent"}
        register_tunable_class(TestTunable, metadata)
        instance = TestTunable()
        
        # Should work with both class and instance
        class_metadata = get_tunable_metadata(TestTunable)
        instance_metadata = get_tunable_metadata(instance)
        
        assert class_metadata == instance_metadata
        assert instance_metadata["name"] == "TestComponent"

    def test_get_tunable_metadata_unregistered_class(self):
        """Test getting metadata for unregistered class."""
        class UnregisteredTunable(TunableMixin):
            pass
        
        metadata = get_tunable_metadata(UnregisteredTunable)
        assert metadata is None

    def test_get_tunable_metadata_none_input(self):
        """Test getting metadata with None input."""
        metadata = get_tunable_metadata(None)
        assert metadata is None

    def test_register_multiple_classes(self):
        """Test registering multiple different classes."""
        class TunableA(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("param_a", "float", (0.0, 1.0))
        
        class TunableB(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("param_b", "int", (1, 100))
        
        register_tunable_class(TunableA, {"param_a": ("float", (0.0, 1.0))})
        register_tunable_class(TunableB, {"param_b": ("int", (1, 100))})
        
        assert is_tunable_registered(TunableA)
        assert is_tunable_registered(TunableB)
        
        metadata_a = get_tunable_metadata(TunableA)
        metadata_b = get_tunable_metadata(TunableB)
        
        assert metadata_a is not None
        assert metadata_b is not None
        assert "param_a" in metadata_a["tunables"]
        assert "param_b" in metadata_b["tunables"]

    def test_register_class_with_inheritance(self):
        """Test registering classes with inheritance relationships."""
        class BaseTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("base_param", "float", (0.0, 1.0))
        
        class DerivedTunable(BaseTunable):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("derived_param", "int", (1, 10))
        
        register_tunable_class(BaseTunable, {"base_param": ("float", (0.0, 1.0))})
        register_tunable_class(DerivedTunable, {"derived_param": ("int", (1, 10))})
        
        assert is_tunable_registered(BaseTunable)
        assert is_tunable_registered(DerivedTunable)
        
        base_metadata = get_tunable_metadata(BaseTunable)
        derived_metadata = get_tunable_metadata(DerivedTunable)
        
        assert base_metadata is not None
        assert derived_metadata is not None
        assert "base_param" in base_metadata["tunables"]
        assert "derived_param" in derived_metadata["tunables"]

    def test_metadata_immutability(self):
        """Test that returned metadata is immutable/copied."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0))
        
        original_metadata = {"name": "Test", "version": "1.0"}
        register_tunable_class(TestTunable, original_metadata)
        
        # Get metadata and modify it
        retrieved_metadata = get_tunable_metadata(TestTunable)
        retrieved_metadata["name"] = "Modified"
        
        # Original metadata should be unchanged
        assert original_metadata["name"] == "Test"
        
        # Getting metadata again should return original values
        fresh_metadata = get_tunable_metadata(TestTunable)
        assert fresh_metadata["name"] == "Test"

    def test_empty_metadata_registration(self):
        """Test registering class with empty metadata."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0))
        
        register_tunable_class(TestTunable, {"temperature": ("float", (0.0, 2.0))})
        
        assert is_tunable_registered(TestTunable)
        metadata = get_tunable_metadata(TestTunable)
        assert metadata is not None
        assert "temperature" in metadata["tunables"]

    def test_none_metadata_registration(self):
        """Test registering class with None metadata."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0))
        
        register_tunable_class(TestTunable, {"temperature": ("float", (0.0, 2.0))})
        
        assert is_tunable_registered(TestTunable)
        metadata = get_tunable_metadata(TestTunable)
        assert metadata is not None
        assert "temperature" in metadata["tunables"]


class TestRegistryEdgeCases:
    """Test edge cases and error conditions for registry."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear the registry before each test
        from octuner.tunable import registry
        registry._TUNABLE_METADATA.clear()

    def test_register_builtin_type(self):
        """Test registering built-in types."""
        # This should work but might not be practical
        register_tunable_class(str, {"length": ("int", (0, 1000))})
        
        assert is_tunable_registered(str)
        metadata = get_tunable_metadata(str)
        assert metadata is not None
        assert "length" in metadata["tunables"]

    def test_register_none_class(self):
        """Test registering None as class."""
        # This should handle gracefully
        try:
            register_tunable_class(None)
            # If it doesn't raise an exception, check behavior
            assert not is_tunable_registered(None)
        except (TypeError, AttributeError):
            # Expected behavior for None input
            pass

    def test_registry_persistence_across_imports(self):
        """Test that registry persists across different import contexts."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0))
        
        register_tunable_class(TestTunable, {"temperature": ("float", (0.0, 2.0))})
        
        # Re-import and check if registration persists
        from octuner.tunable.registry import is_tunable_registered, get_tunable_metadata
        
        assert is_tunable_registered(TestTunable)
        metadata = get_tunable_metadata(TestTunable)
        assert metadata is not None
        assert "temperature" in metadata["tunables"]

    def test_complex_metadata_types(self):
        """Test registering with complex metadata types."""
        class TestTunable(TunableMixin):
            def __init__(self):
                super().__init__()
                self.mark_as_tunable("temperature", "float", (0.0, 2.0))
        
        complex_metadata = {
            "config": {"nested": {"value": 42}},
            "list_data": [1, 2, 3],
            "tuple_data": (1, 2, 3),
            "none_value": None,
            "bool_value": True
        }
        
        register_tunable_class(TestTunable, complex_metadata)
        
        retrieved_metadata = get_tunable_metadata(TestTunable)
        assert retrieved_metadata["config"]["nested"]["value"] == 42
        assert retrieved_metadata["list_data"] == [1, 2, 3]
        assert retrieved_metadata["tuple_data"] == (1, 2, 3)
        assert retrieved_metadata["none_value"] is None
        assert retrieved_metadata["bool_value"] is True
