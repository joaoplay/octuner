import re
from typing import Any, Dict, List, Optional, Set, Tuple
from ..tunable.mixin import is_llm_tunable, get_tunable_parameters, get_proxy_attribute
from ..tunable.types import ParamType


class ComponentDiscovery:
    """
    Discovers tunable components in a component tree.

    Recursively finds all objects implementing TunableMixin protocol
    and builds the search space for optimization.
    """

    def __init__(self, include_patterns: Optional[List[str]] = None, exclude_patterns: Optional[List[str]] = None):
        """
        Initialize discovery with optional filters.
        
        Args:
            include_patterns: Glob patterns to include (e.g., ["*.temperature"])
            exclude_patterns: Glob patterns to exclude (e.g., ["*.verbose"])
        """
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def discover(self, component: Any) -> Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]:
        """
        Discover all tunable components in the component tree.
        
        Args:
            component: Root component to search
            
        Returns:
            Dictionary mapping dotted paths to tunable parameter definitions
        """
        discovered = {}
        visited = set()

        self._discover_recursive(component, "", discovered, visited)

        # Apply filters
        filtered = self._apply_filters(discovered)

        return filtered

    def _discover_recursive(self, obj: Any, path: str, discovered: Dict[str, Dict[str, Tuple[ParamType, Any, Any]]],
                            visited: Set[int]) -> None:
        """
        Recursively discover tunable components.
        
        Args:
            obj: Object to examine
            path: Current dotted path
            discovered: Dictionary to populate with discoveries
            visited: Set of object IDs to avoid cycles
        """
        if obj is None:
            return

        obj_id = id(obj)
        if obj_id in visited:
            return

        visited.add(obj_id)

        # Check if this object is tunable
        if is_llm_tunable(obj):
            tunables = get_tunable_parameters(obj)
            if tunables:
                discovered[path] = tunables

        # Recursively search attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                # Skip private attributes and methods
                if attr_name.startswith('_') or callable(attr_value):
                    continue

                new_path = f"{path}.{attr_name}" if path else attr_name
                self._discover_recursive(attr_value, new_path, discovered, visited)

        # Skip the dir() approach to avoid circular references
        # Only explore __dict__ attributes which are safer

    def _apply_filters(self, discovered: Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]) -> Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]:
        """
        Apply "include/exclude" filters to discovered components.
        
        Args:
            discovered: Raw discovery results
            
        Returns:
            Filtered discovery results
        """
        if not self.include_patterns and not self.exclude_patterns:
            return discovered

        filtered = {}

        for path, tunables in discovered.items():
            # Apply filters to individual p arameters
            filtered_tunables = {}
            for param_name, param_def in tunables.items():
                param_path = f"{path}.{param_name}"

                # Check parameter-level filters
                if self.include_patterns:
                    if not any(self._matches_pattern(param_path, pattern) for pattern in self.include_patterns):
                        continue

                if self.exclude_patterns:
                    if any(self._matches_pattern(param_path, pattern) for pattern in self.exclude_patterns):
                        continue

                filtered_tunables[param_name] = param_def

            if filtered_tunables:
                filtered[path] = filtered_tunables

        return filtered

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches a glob pattern.
        
        Args:
            path: Dotted path to check
            pattern: Glob pattern (e.g., "*.temperature")
            
        Returns:
            True if path matches pattern
        """
        # Convert glob pattern to regex
        regex_pattern = pattern.replace('.', r'\.').replace('*', r'.*')
        return re.match(regex_pattern, path) is not None


def discover_tunable_components(component: Any, include_patterns: Optional[List[str]] = None,
                                exclude_patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]:
    """
    Convenience function to discover tunable components.
    
    Args:
        component: Root component to search
        include_patterns: Glob patterns to include
        exclude_patterns: Glob patterns to exclude
        
    Returns:
        Dictionary mapping dotted paths to tunable parameter definitions
    """
    discovery = ComponentDiscovery(include_patterns, exclude_patterns)
    return discovery.discover(component)


def build_search_space(discovered: Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]) -> Dict[str, Tuple[ParamType, Any, Any]]:
    """
    Build the search space from discovered components.
    
    Args:
        discovered: Discovery results from ComponentDiscovery
        
    Returns:
        Dictionary mapping full parameter paths to parameter definitions
    """
    search_space = {}

    for component_path, tunables in discovered.items():
        for param_name, param_def in tunables.items():
            full_path = f"{component_path}.{param_name}"
            search_space[full_path] = param_def

    return search_space
