"""
Parameter setting for Octuner.

Safely sets parameter values on discovered components,
handling proxy attributes and type validation.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union
from ..tunable.mixin import get_proxy_attribute, is_llm_tunable, get_tunable_parameters
from ..tunable.types import ParamType

logger = logging.getLogger(__name__)


class ParameterSetter:
    """
    Sets parameter values on discovered components.
    """
    
    def __init__(self, component: Any):
        """
        Initialize with the root component.
        
        Args:
            component: Root component to set parameters on
        """
        self.component = component
        self._cache = {}  # Cache for resolved object paths
    
    def set_parameters(self, parameters: Dict[str, Any], strict: bool = False) -> None:
        """
        Set multiple parameters at once.
        
        Args:
            parameters: Dictionary mapping dotted paths to values
            strict: If True, raise exceptions instead of logging warnings
        """
        for path, value in parameters.items():
            try:
                self.set_parameter(path, value)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Failed to set parameter {path}: {e}")
    
    def set_parameter(self, path: str, value: Any) -> None:
        """
        Set a single parameter by dotted path.
        
        Args:
            path: Dotted path to the parameter (e.g., "llm.temperature")
            value: Value to set
            
        Raises:
            ValueError: If path is invalid or value type is wrong
            AttributeError: If path doesn't exist
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Parse the path
        parts = path.split('.')
        if len(parts) < 1:
            raise ValueError(f"Invalid path format: {path}")
        
        # Handle direct attributes on the root component
        if len(parts) == 1:
            obj = self.component
            param_name = parts[0]
        else:
            # Navigate to the component
            component_path = '.'.join(parts[:-1])
            param_name = parts[-1]
            obj = self._resolve_object(component_path)
            if obj is None:
                raise AttributeError(f"Component path not found: {component_path}")
        
        # Check if this object is tunable
        if not is_llm_tunable(obj):
            if len(parts) == 1:
                raise AttributeError(f"Root component is not tunable")
            else:
                raise AttributeError(f"Object at {component_path} is not tunable")
        
        # Get parameter definition
        tunables = get_tunable_parameters(obj)
        if param_name not in tunables:
            if len(parts) == 1:
                raise AttributeError(f"Parameter {param_name} not found in root component")
            else:
                raise AttributeError(f"Parameter {param_name} not found in {component_path}")
        
        param_def = tunables[param_name]
        param_type = param_def[0]
        
        # Validate and coerce value
        # Handle different parameter definition formats
        if param_type in ["float", "int"]:
            # For numeric types, param_def[1] might be a tuple (min, max) or just min
            if isinstance(param_def[1], tuple) and len(param_def[1]) == 2:
                low, high = param_def[1]
            else:
                low = param_def[1]
                high = param_def[2] if len(param_def) > 2 else None
        else:
            # For choice/bool types, param_def[1] contains the choices
            low = param_def[1]
            high = param_def[2] if len(param_def) > 2 else None
        
        coerced_value = self._validate_and_coerce_value(value, param_type, low, high)
        
        # Set the value (possibly on a proxy object)
        proxy_attr = get_proxy_attribute(obj)
        if proxy_attr:
            # Set on the nested object
            proxy_obj = getattr(obj, proxy_attr, None)
            if proxy_obj is None:
                raise AttributeError(f"Proxy attribute {proxy_attr} not found on {component_path}")
            setattr(proxy_obj, param_name, coerced_value)
        else:
            # Set directly on the object
            setattr(obj, param_name, coerced_value)
        
        logger.debug(f"Set {path} = {coerced_value}")
    
    def get_parameter(self, path: str) -> Any:
        """
        Get a parameter value by dotted path.
        
        Args:
            path: Dotted path to the parameter
            
        Returns:
            Current parameter value
            
        Raises:
            AttributeError: If path doesn't exist
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        parts = path.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid path format: {path}")
        
        component_path = '.'.join(parts[:-1])
        param_name = parts[-1]
        
        obj = self._resolve_object(component_path)
        if obj is None:
            raise AttributeError(f"Component path not found: {component_path}")
        
        proxy_attr = get_proxy_attribute(obj)
        if proxy_attr:
            proxy_obj = getattr(obj, proxy_attr, None)
            if proxy_obj is None:
                raise AttributeError(f"Proxy attribute {proxy_attr} not found on {component_path}")
            return getattr(proxy_obj, param_name)
        else:
            return getattr(obj, param_name)
    
    def _resolve_object(self, path: str) -> Optional[Any]:
        """
        Resolve a dotted path to an object.
        
        Args:
            path: Dotted path to resolve
            
        Returns:
            Resolved object, or None if not found
        """
        if not path:
            return self.component
        
        # Check cache first
        if path in self._cache:
            return self._cache[path]
        
        parts = path.split('.')
        current = self.component
        
        for part in parts:
            if not hasattr(current, part):
                return None
            
            current = getattr(current, part)
            if current is None:
                return None
        
        # Cache the result
        self._cache[path] = current
        return current
    
    def _validate_and_coerce_value(
        self, 
        value: Any, 
        param_type: ParamType, 
        low: Any, 
        high: Any
    ) -> Any:
        """
        Validate and coerce a parameter value.
        
        Args:
            value: Value to validate/coerce
            param_type: Expected parameter type
            low: Lower bound or choices
            high: Upper bound
            
        Returns:
            Coerced value
            
        Raises:
            ValueError: If value is invalid for the parameter type
        """
        if param_type == "float":
            try:
                coerced = float(value)
                if low is not None and coerced < low:
                    raise ValueError(f"Value {coerced} below minimum {low}")
                if high is not None and coerced > high:
                    raise ValueError(f"Value {coerced} above maximum {high}")
                return coerced
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert {value} to float")
        
        elif param_type == "int":
            if value is None:
                return None
            try:
                coerced = int(value)
                if low is not None and coerced < low:
                    raise ValueError(f"Value {coerced} below minimum {low}")
                if high is not None and coerced > high:
                    raise ValueError(f"Value {coerced} above maximum {high}")
                return coerced
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert {value} to int")
        
        elif param_type == "choice":
            if value not in low:  # low contains choices for choice type
                raise ValueError(f"Value {value} not in choices {low}")
            return value
        
        elif param_type == "bool":
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                if value.lower() in ('true', '1', 'yes', 'on'):
                    return True
                elif value.lower() in ('false', '0', 'no', 'off'):
                    return False
                else:
                    raise ValueError(f"Cannot convert string '{value}' to bool")
            elif isinstance(value, (int, float)):
                return bool(value)
            else:
                raise ValueError(f"Cannot convert {type(value).__name__} to bool")
        
        elif param_type == "list":
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                # Try to parse as JSON list
                import json
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # If not JSON, split by comma and strip whitespace
                    return [item.strip() for item in value.split(',') if item.strip()]
            else:
                return list(value) if value else []
        
        
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")


def set_parameters(component: Any, parameters: Dict[str, Any], strict: bool = False) -> None:
    """
    Convenience function to set parameters on a component.
    
    Args:
        component: Component to set parameters on
        parameters: Dictionary mapping dotted paths to values
        strict: If True, raise exceptions instead of logging warnings
    """
    setter = ParameterSetter(component)
    setter.set_parameters(parameters, strict)


def get_parameter(component: Any, path: str) -> Any:
    """
    Convenience function to get a parameter value.
    
    Args:
        component: Component to get parameter from
        path: Dotted path to the parameter
        
    Returns:
        Parameter value
    """
    setter = ParameterSetter(component)
    return setter.get_parameter(path)
