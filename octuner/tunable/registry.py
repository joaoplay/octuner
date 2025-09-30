from typing import Any, Dict, Optional, Tuple
from .types import ParamType

# Metadata storage for tunable classes
_TUNABLE_METADATA = {}


def _ensure_metadata(cls) -> None:
    """
    Ensure metadata dictionary exists for a class.

    Args:
        cls: Class to ensure metadata for
    """
    if cls not in _TUNABLE_METADATA:
        _TUNABLE_METADATA[cls] = {
            'tunables': {},
            'call_method': 'send_prompt',
            'proxy': None
        }


def get_tunable_metadata(cls_or_obj) -> Optional[Dict[str, Any]]:
    """
    Get tunable metadata for a class or object.
    
    Args:
        cls_or_obj: Class or instance to get metadata for
        
    Returns:
        Dictionary with metadata or None if not registered
    """
    if cls_or_obj is None:
        return None
        
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)

    # Check for registered metadata
    if cls in _TUNABLE_METADATA:
        return _TUNABLE_METADATA[cls].copy()

    # Return None if not registered
    return None


def is_tunable_registered(cls_or_obj) -> bool:
    """
    Check if a class or object has been registered as tunable.
    
    Args:
        cls_or_obj: Class or instance to check
        
    Returns:
        True if the object has been registered and has tunable parameters
    """
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)

    # Check if the class has been registered and has tunable parameters
    if cls in _TUNABLE_METADATA:
        metadata = _TUNABLE_METADATA[cls]
        return bool(metadata.get('tunables', {}))
    return False


def register_tunable_class(cls, metadata_or_params: Dict[str, Any] = None, call_method: str = "send_prompt",
                           proxy: Optional[str] = None) -> None:
    """
    Programmatically register a class as tunable.
    
    This is used when tunable parameters need to be determined at runtime,
    such as when the available models depend on configuration.
    
    Args:
        cls: Class to register
        metadata_or_params: Dictionary of tunable parameters or arbitrary metadata
        call_method: Method name to wrap for timing/cost measurement
        proxy: Proxy attribute name (if any)
    """
    if metadata_or_params is None:
        # Only add defaults when no metadata is provided
        _TUNABLE_METADATA[cls] = {
            'tunables': {},
            'call_method': call_method,
            'proxy': proxy
        }
    else:
        # Check if this looks like tunable parameters (values are tuples) vs arbitrary metadata
        looks_like_tunables = all(
            isinstance(v, tuple) and len(v) >= 2 
            for v in metadata_or_params.values() 
            if v is not None
        ) if metadata_or_params else False
        
        if looks_like_tunables:
            # Store as tunable parameters under 'tunables' key
            _TUNABLE_METADATA[cls] = {
                'tunables': metadata_or_params.copy(),
                'call_method': call_method,
                'proxy': proxy
            }
        else:
            # Store as arbitrary metadata
            _TUNABLE_METADATA[cls] = metadata_or_params.copy()
            
            # Only add required keys if they're not present and we have non-empty metadata
            # or if they were explicitly provided as non-default values
            if call_method != "send_prompt" and 'call_method' not in _TUNABLE_METADATA[cls]:
                _TUNABLE_METADATA[cls]['call_method'] = call_method
            if proxy is not None and 'proxy' not in _TUNABLE_METADATA[cls]:
                _TUNABLE_METADATA[cls]['proxy'] = proxy


# Export the essential registry functions
__all__ = [
    'get_tunable_metadata',
    'is_tunable_registered',
    'register_tunable_class',
]
