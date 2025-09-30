"""
Unit tests for TunableMixin functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from octuner.tunable.mixin import TunableMixin, is_llm_tunable, get_tunable_parameters


class TestTunableMixin:
    """Test TunableMixin class functionality."""

    def test_tunable_mixin_init(self):
        """Test TunableMixin initialization."""
        mixin = TunableMixin()
        assert hasattr(mixin, '_tunable_params')
        assert mixin._tunable_params == {}

    def test_mark_as_tunable_basic(self):
        """Test marking parameters as tunable with basic usage."""
        mixin = TunableMixin()
        
        # Mark single parameter as tunable
        mixin.mark_as_tunable("temperature", param_type="float", range_vals=(0.0, 2.0))
        
        assert "temperature" in mixin._tunable_params
        param_info = mixin._tunable_params["temperature"]
        assert param_info["type"] == "float"
        assert param_info["range"] == (0.0, 2.0)
        assert param_info["default"] is None

    def test_mark_as_tunable_with_default(self):
        """Test marking parameters as tunable with default value."""
        mixin = TunableMixin()
        
        mixin.mark_as_tunable(
            "max_tokens", 
            param_type="int", 
            range_vals=(50, 4000),
            default=1000
        )
        
        param_info = mixin._tunable_params["max_tokens"]
        assert param_info["type"] == "int"
        assert param_info["range"] == (50, 4000)
        assert param_info["default"] == 1000

    def test_mark_as_tunable_choice_type(self):
        """Test marking choice parameters as tunable."""
        mixin = TunableMixin()
        
        mixin.mark_as_tunable(
            "model", 
            param_type="choice", 
            range_vals=("gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"),
            default="gpt-4o-mini"
        )
        
        param_info = mixin._tunable_params["model"]
        assert param_info["type"] == "choice"
        assert param_info["range"] == ("gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini")
        assert param_info["default"] == "gpt-4o-mini"

    def test_mark_as_tunable_bool_type(self):
        """Test marking boolean parameters as tunable."""
        mixin = TunableMixin()
        
        mixin.mark_as_tunable(
            "use_websearch", 
            param_type="bool", 
            range_vals=(True, False),
            default=False
        )
        
        param_info = mixin._tunable_params["use_websearch"]
        assert param_info["type"] == "bool"
        assert param_info["range"] == (True, False)
        assert param_info["default"] is False

    def test_mark_as_tunable_multiple_params(self):
        """Test marking multiple parameters as tunable."""
        mixin = TunableMixin()
        
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        mixin.mark_as_tunable("top_p", "float", (0.0, 1.0), 1.0)
        mixin.mark_as_tunable("max_tokens", "int", (50, 4000), 1000)
        
        assert len(mixin._tunable_params) == 3
        assert "temperature" in mixin._tunable_params
        assert "top_p" in mixin._tunable_params
        assert "max_tokens" in mixin._tunable_params

    def test_mark_as_tunable_override(self):
        """Test overriding existing tunable parameter."""
        mixin = TunableMixin()
        
        # First definition
        mixin.mark_as_tunable("temperature", "float", (0.0, 1.0), 0.5)
        assert mixin._tunable_params["temperature"]["range"] == (0.0, 1.0)
        
        # Override with new range
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        assert mixin._tunable_params["temperature"]["range"] == (0.0, 2.0)
        assert mixin._tunable_params["temperature"]["default"] == 0.7

    def test_get_tunable_parameters(self):
        """Test getting tunable parameters."""
        mixin = TunableMixin()
        
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        mixin.mark_as_tunable("max_tokens", "int", (50, 4000), 1000)
        
        params = mixin.get_tunable_parameters()
        assert len(params) == 2
        assert "temperature" in params
        assert "max_tokens" in params
        assert params["temperature"]["type"] == "float"
        assert params["max_tokens"]["type"] == "int"

    def test_get_tunable_parameters_empty(self):
        """Test getting tunable parameters when none are defined."""
        mixin = TunableMixin()
        params = mixin.get_tunable_parameters()
        assert params == {}

    def test_get_tunable_parameters_immutable(self):
        """Test that returned parameters dictionary is a copy."""
        mixin = TunableMixin()
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        params = mixin.get_tunable_parameters()
        
        # Modify returned dictionary
        params["new_param"] = {"type": "int"}
        
        # Original should be unchanged
        assert "new_param" not in mixin._tunable_params
        assert len(mixin._tunable_params) == 1

    def test_is_tunable(self):
        """Test checking if parameter is tunable."""
        mixin = TunableMixin()
        
        assert not mixin.is_tunable("temperature")
        
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0))
        assert mixin.is_tunable("temperature")
        assert not mixin.is_tunable("nonexistent_param")

    def test_get_param_info_existing(self):
        """Test getting parameter info for existing parameter."""
        mixin = TunableMixin()
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        param_info = mixin.get_param_info("temperature")
        assert param_info is not None
        assert param_info["type"] == "float"
        assert param_info["range"] == (0.0, 2.0)
        assert param_info["default"] == 0.7

    def test_get_param_info_nonexistent(self):
        """Test getting parameter info for non-existent parameter."""
        mixin = TunableMixin()
        param_info = mixin.get_param_info("nonexistent")
        assert param_info is None

    def test_get_param_info_immutable(self):
        """Test that returned parameter info is a copy."""
        mixin = TunableMixin()
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        
        param_info = mixin.get_param_info("temperature")
        
        # Modify returned info
        param_info["type"] = "int"
        
        # Original should be unchanged
        original_info = mixin._tunable_params["temperature"]
        assert original_info["type"] == "float"


class TestTunableMixinInheritance:
    """Test TunableMixin when used with inheritance."""

    def test_mixin_with_custom_class(self):
        """Test TunableMixin with custom class."""
        class CustomLLMComponent(TunableMixin):
            def __init__(self):
                super().__init__()
                self.temperature = 0.7
                self.max_tokens = 1000
                
                # Mark parameters as tunable
                self.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
                self.mark_as_tunable("max_tokens", "int", (50, 4000), 1000)
            
            def generate(self, prompt):
                return f"Generated with temp={self.temperature}, tokens={self.max_tokens}"
        
        component = CustomLLMComponent()
        assert component.is_tunable("temperature")
        assert component.is_tunable("max_tokens")
        assert not component.is_tunable("nonexistent")
        
        params = component.get_tunable_parameters()
        assert len(params) == 2

    def test_multiple_inheritance(self):
        """Test TunableMixin with multiple inheritance."""
        class BaseComponent:
            def __init__(self):
                self.name = "base"
        
        class TunableComponent(BaseComponent, TunableMixin):
            def __init__(self):
                BaseComponent.__init__(self)
                TunableMixin.__init__(self)
                self.mark_as_tunable("param1", "float", (0.0, 1.0))
        
        component = TunableComponent()
        assert component.name == "base"
        assert component.is_tunable("param1")


class TestUtilityFunctions:
    """Test utility functions for tunable detection."""

    def test_is_llm_tunable_with_tunable_object(self):
        """Test is_llm_tunable with tunable object."""
        mixin = TunableMixin()
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0))
        
        assert is_llm_tunable(mixin) is True

    def test_is_llm_tunable_with_non_tunable_object(self):
        """Test is_llm_tunable with non-tunable object."""
        class RegularClass:
            pass
        
        obj = RegularClass()
        assert is_llm_tunable(obj) is False

    def test_is_llm_tunable_with_empty_tunable(self):
        """Test is_llm_tunable with tunable object that has no parameters."""
        mixin = TunableMixin()
        # No parameters marked as tunable
        
        assert is_llm_tunable(mixin) is False

    def test_is_llm_tunable_with_none(self):
        """Test is_llm_tunable with None."""
        assert is_llm_tunable(None) is False

    def test_get_tunable_parameters_function_with_tunable_object(self):
        """Test get_tunable_parameters function with tunable object."""
        mixin = TunableMixin()
        mixin.mark_as_tunable("temperature", "float", (0.0, 2.0), 0.7)
        mixin.mark_as_tunable("max_tokens", "int", (50, 4000), 1000)
        
        params = get_tunable_parameters(mixin)
        assert len(params) == 2
        assert "temperature" in params
        assert "max_tokens" in params

    def test_get_tunable_parameters_function_with_non_tunable_object(self):
        """Test get_tunable_parameters function with non-tunable object."""
        class RegularClass:
            pass
        
        obj = RegularClass()
        params = get_tunable_parameters(obj)
        assert params == {}

    def test_get_tunable_parameters_function_with_none(self):
        """Test get_tunable_parameters function with None."""
        params = get_tunable_parameters(None)
        assert params == {}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_mark_as_tunable_invalid_param_type(self):
        """Test marking parameter with invalid type."""
        mixin = TunableMixin()
        
        # Note: The current implementation doesn't validate param_type,
        # so this will actually work. This test documents current behavior.
        mixin.mark_as_tunable("param", "invalid_type", (0, 1))
        
        param_info = mixin.get_param_info("param")
        assert param_info["type"] == "invalid_type"

    def test_mark_as_tunable_empty_range(self):
        """Test marking parameter with empty range."""
        mixin = TunableMixin()
        
        # Empty tuple as range
        mixin.mark_as_tunable("param", "choice", ())
        
        param_info = mixin.get_param_info("param")
        assert param_info["range"] == ()

    def test_mark_as_tunable_none_range(self):
        """Test marking parameter with None range."""
        mixin = TunableMixin()
        
        mixin.mark_as_tunable("param", "float", None)
        
        param_info = mixin.get_param_info("param")
        assert param_info["range"] is None

    def test_tunable_params_attribute_direct_access(self):
        """Test direct access to _tunable_params attribute."""
        mixin = TunableMixin()
        
        # Direct modification (not recommended but should work)
        mixin._tunable_params["direct_param"] = {
            "type": "float",
            "range": (0.0, 1.0),
            "default": 0.5
        }
        
        assert mixin.is_tunable("direct_param")
        param_info = mixin.get_param_info("direct_param")
        assert param_info["type"] == "float"

    def test_mark_as_tunable_with_complex_objects(self):
        """Test marking parameters with complex objects."""
        mixin = TunableMixin()
        
        # Using complex objects as range values
        complex_range = ({"model": "gpt-3.5"}, {"model": "gpt-4"})
        mixin.mark_as_tunable("model_config", "choice", complex_range)
        
        param_info = mixin.get_param_info("model_config")
        assert param_info["range"] == complex_range
