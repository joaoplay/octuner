from typing import Any, Dict, Optional, Tuple
from .types import ParamType
from .registry import get_tunable_metadata, is_tunable_registered


class TunableMixin:
    """
    Mixin class for LLM components that can be auto-tuned.
    
    Components can be made tunable by either:
    1. Using instance methods like mark_as_tunable() (legacy approach)
    2. Programmatic registration using register_tunable_class() (recommended)
    
    Example:
        class MyLLM(TunableMixin):
            def __init__(self):
                super().__init__()
                # Legacy approach
                self.mark_as_tunable("temperature", "float", (0.0, 1.0), 0.7)
                
                # Or programmatic registration (recommended)
                from octuner.tunable.registry import register_tunable_class
                register_tunable_class(
                    self.__class__,
                    params={
                        "temperature": ("float", 0.0, 1.0),
                        "max_tokens": ("int", 64, 4096),
                    },
                    call_method="send_prompt"
                )
    """

    def __init__(self):
        """Initialize the tunable mixin."""
        self._tunable_params = {}

    def mark_as_tunable(self, param_name: str, param_type: str, range_vals: Tuple[Any, Any], 
                       default: Any = None) -> None:
        """
        Mark a parameter as tunable.
        
        Args:
            param_name: Name of the parameter
            param_type: Type of parameter ("float", "int", "choice", "bool")
            range_vals: Range tuple (min, max) for numeric types, choices for choice type
            default: Default value for the parameter
        """
        self._tunable_params[param_name] = {
            "type": param_type,
            "range": range_vals,
            "default": default
        }

    def get_tunable_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tunable parameters.
        
        Returns:
            Dictionary of tunable parameter definitions
        """
        return self._tunable_params.copy()

    def is_tunable(self, param_name: str) -> bool:
        """
        Check if a parameter is tunable.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            True if the parameter is tunable
        """
        return param_name in self._tunable_params

    def get_param_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tunable parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Parameter info dictionary or None if not found
        """
        param_info = self._tunable_params.get(param_name)
        if param_info is not None:
            return param_info.copy()
        return None

    def llm_eq_cost(self, *, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Calculate the cost of an LLM call (optional).
        
        Override this method to enable cost tracking during optimization.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens  
            metadata: Additional metadata from the LLM call
            
        Returns:
            Cost in your preferred currency, or None to disable cost tracking
        """
        return None


def is_llm_tunable(obj: Any) -> bool:
    """
    Check if an object is tunable (registered as tunable or has instance tunables).
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object has tunable configuration
    """
    if obj is None:
        return False
    
    # Check for instance-based tunables (legacy)
    if hasattr(obj, 'get_tunable_parameters'):
        instance_params = obj.get_tunable_parameters()
        if instance_params:
            return True
    
    # Check for registry-based tunables
    return is_tunable_registered(obj)


def get_tunable_parameters(obj: Any) -> Dict[str, Tuple[ParamType, Any, Any]]:
    """
    Get tunable parameters from a tunable object.
    
    Supports both instance-based (legacy) and registry-based approaches.
    
    Args:
        obj: Object with tunable configuration
        
    Returns:
        Dictionary of parameter definitions (empty dict if not tunable)
    """
    if obj is None:
        return {}
    
    # First try instance-based approach (legacy)
    if hasattr(obj, 'get_tunable_parameters'):
        instance_params = obj.get_tunable_parameters()
        if instance_params:
            # Convert to expected format
            result = {}
            for param_name, param_info in instance_params.items():
                if isinstance(param_info, dict):
                    # Dictionary format: {"type": "float", "range": (0.0, 2.0), "default": 0.7}
                    param_type = param_info["type"]
                    range_vals = param_info["range"]
                    default = param_info.get("default")
                    
                    # For numeric types, extract min/max from range tuple
                    if param_type in ['float', 'int'] and isinstance(range_vals, tuple) and len(range_vals) == 2:
                        result[param_name] = (param_type, range_vals[0], range_vals[1])
                    else:
                        # For choice, list, bool types, use the range as choices
                        result[param_name] = (param_type, range_vals, default)
                elif isinstance(param_info, tuple):
                    # Tuple format: ("float", 0.0, 2.0)
                    result[param_name] = param_info
                else:
                    # Unknown format, skip
                    continue
            return result
    
    # Fall back to registry-based approach
    if not is_tunable_registered(obj):
        return {}

    metadata = get_tunable_metadata(obj)
    return metadata.get('tunables', {}).copy() if metadata else {}


def get_call_method_name(obj: Any) -> str:
    """
    Get the method name to wrap for timing/cost measurement.
    
    Args:
        obj: Object with tunable configuration
        
    Returns:
        Method name to wrap (defaults to "send_prompt")
    """
    if not is_llm_tunable(obj):
        return "send_prompt"

    metadata = get_tunable_metadata(obj)
    return metadata.get('call_method', 'send_prompt') if metadata else 'send_prompt'


def get_proxy_attribute(obj: Any) -> Optional[str]:
    """
    Get the proxy attribute name if tunables live on a nested object.
    
    Args:
        obj: Object with tunable configuration
        
    Returns:
        Proxy attribute name, or None if not set
    """
    if not is_llm_tunable(obj):
        return None

    metadata = get_tunable_metadata(obj)
    return metadata.get('proxy') if metadata else None
