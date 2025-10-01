"""
Octuner - Auto-tuning library for multi-call LLM components.

A lightweight library that optimizes parameters of multi-call LLM components
via auto-discovery and tuning against golden datasets.
"""


from .tunable.mixin import TunableMixin, is_llm_tunable, get_tunable_parameters
from .tunable.registry import (
    get_tunable_metadata, is_tunable_registered, 
    register_tunable_class
)
from .optimization.auto import AutoTuner
from .utils.exporter import apply_best
from .tunable.types import SearchResult, ParamType
from .tunable.tunable_llm import MultiProviderTunableLLM
from .providers import (
    get_provider, 
    get_all_models, 
    register_provider,
    unregister_provider,
    list_providers,
    PROVIDERS
)

__version__ = "0.1.0"

__all__ = [
    "TunableMixin", 
    "is_llm_tunable",
    "get_tunable_parameters",
    # Registry functions
    "get_tunable_metadata",
    "is_tunable_registered",
    "register_tunable_class",
    # Services
    "AutoTuner", 
    "apply_best", 
    "SearchResult", 
    "ParamType",
    "MultiProviderTunableLLM",
    # Provider functions
    "get_provider",
    "get_all_models",
    "register_provider",
    "unregister_provider",
    "list_providers",
    "PROVIDERS",
]
