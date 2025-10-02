import re
from typing import Any, Dict, List, Optional, Set, Tuple
from ..tunable.mixin import is_llm_tunable, get_tunable_parameters, get_proxy_attribute
from ..tunable.types import ParamType


class ComponentDiscovery:
    """
    This is a core component of Octuner that automatically finds and catalogs all tunable
    parameters within an object hierarchy. It recursively traverses object attributes to
    identify components that implement the TunableMixin protocol, building a search space
    that can be optimized by the AutoTuner.

    How it works:
    1. **Recursive Traversal**: Starting from a root component, it recursively explores
        all object attributes using `__dict__` introspection, avoiding circular references
        and method calls.

    2. **Tunable Detection**: For each object found, it checks if it implements the
        TunableMixin protocol using `is_llm_tunable()`, which supports both:
            - Instance-based tunables (legacy): Objects with `get_tunable_parameters()`
                                                method
            - Registry-based tunables (recommended): Objects registered through
                                                     `register_tunable_class()`

    3. **Parameter Extraction**: For tunable objects, it extracts parameter definitions
        including type, range, and default values using `get_tunable_parameters()`.

    4. **Path Mapping**: Each discovered parameter is mapped to a dotted path
        (e.g., "classifier_llm.temperature") that uniquely identifies its location in
        the hierarchy.

    5. **Filtering**: Optional include/exclude patterns can be used to focus the search
        space on specific parameters or components.

    Example:
        ```python
        from octuner import ComponentDiscovery, MultiProviderTunableLLM
        
        # Create a complex component hierarchy
        analyzer = TunableSentimentAnalyzer(config_file)
        
        # Discover all tunable parameters
        discovery = ComponentDiscovery()
        tunables = discovery.discover(analyzer)
        
        # Result: {
        #     "classifier_llm": {
        #         "temperature": ("float", 0.0, 2.0),
        #         "max_tokens": ("int", 64, 4096),
        #         "provider_model": ("choice", ["openai:gpt-4", "gemini:gemini-pro"])
        #     },
        #     "confidence_llm": { ... },
        #     "reasoning_llm": { ... }
        # }
        
        # Focus on specific parameters
        focused_discovery = ComponentDiscovery(
            include_patterns=["*.temperature", "*.provider_model"],
            exclude_patterns=["*.verbose"]
        )
        focused_tunables = focused_discovery.discover(analyzer)
        ```

    Integration with AutoTuner:
        ComponentDiscovery is automatically used by AutoTuner to build the search space
        before optimization begins. The discovered parameters become the dimensions that
         the optimizer explores to find the best configuration.

    Attributes:
        include_patterns: List of glob patterns to include in discovery
        exclude_patterns: List of glob patterns to exclude from discovery
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

        This is the main entry point for component discovery. It performs a recursive
        traversal of the component hierarchy, identifying all objects that implement
        the TunableMixin protocol and extracting their tunable parameters.

        The discovery process:
            1. Starts from the provided root component
            2. Recursively explores all object attributes
            3. Identifies tunable components using `is_llm_tunable()`
            4. Extracts parameter definitions from each tunable component
            5. Applies "include/exclude" filters (optional)
            6. Returns a structured mapping of paths to parameters

        Args:
            component: Root component to search. Can be any Python object, but typically
                     a component hierarchy containing multiple tunable LLMs or other
                     tunable components.

        Returns:
            Dictionary mapping dotted paths to tunable parameter definitions. The structure is:
            ```
            {
                "component_path": {
                    "param_name": (param_type, min_value, max_value),
                    "another_param": (param_type, choices_or_default, ...)
                }
            }
            ```
            
            Where:
            - `component_path`: Dotted path to the component (e.g., "classifier_llm")
            - `param_name`: Name of the tunable parameter
            - `param_type`: Type of parameter ("float", "int", "choice", "bool")
            - `min_value`, `max_value`: Range for numeric parameters
            - `choices_or_default`: Choices for choice parameters or default values

        Note:
            I suppose this method is safe to call on any object. It will return an empty
            dictionary if no tunable components are found.
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
        Recursively discover tunable components in the object hierarchy.

        This is the core recursive traversal method that explores object attributes
        to find tunable components. It implements several safety mechanisms to
        prevent infinite loops and avoid side effects.

        Process:
            1. Check if object is None (base case)
            2. Check if object has been visited (cycle detection)
            3. Mark object as visited
            4. Check if object is tunable using `is_llm_tunable()`
            5. If tunable, extract parameters using `get_tunable_parameters()`
            6. Recursively explore all non-private, non-callable attributes

        Args:
            obj: Object to examine. Can be any Python object.
            path: Current dotted path to this object (e.g., "parent.child")
            discovered: Dictionary to populate with discovered tunable components.
                       Modified in-place during traversal.
            visited: Set of object IDs that have been visited. Used to prevent
                    infinite loops in circular reference scenarios.

        Note:
            This method modifies the `discovered` and `visited` dictionaries/sets
            in-place as it traverses the object hierarchy. The `visited` set is
            used to prevent infinite recursion when objects contain circular
            references.

        Implementation details:
            - Uses `id(obj)` to generate unique identifiers for cycle detection
            - Only explores `obj.__dict__` to avoid calling methods or accessing
              properties that might have side effects
            - Skips attributes starting with '_' to focus on public configuration
            - Skips callable attributes to avoid executing methods during discovery
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

    def _apply_filters(self, discovered: Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]) -> Dict[
        str, Dict[str, Tuple[ParamType, Any, Any]]]:
        """
        Apply "include/exclude" filters to discovered components.

        This method filters the raw discovery results based on the include_patterns
        and exclude_patterns configured during ComponentDiscovery initialization.
        It supports glob-style pattern matching for fine-grained control over
        which parameters are included in the final search space.

        Pattern examples:
        - `"*.temperature"`: Matches any parameter named "temperature" in any component
        - `"classifier_llm.*"`: Matches all parameters in the "classifier_llm" component
        - `"*.provider_model"`: Matches "provider_model" in any component
        - `"*.verbose"`: Matches "verbose" parameters (often used for exclusion)

        Args:
            discovered: Raw discovery results from `_discover_recursive()`. Dictionary
                       mapping component paths to their tunable parameters.

        Returns:
            Filtered discovery results. Dictionary with the same structure as input,
            but containing only parameters that pass the include/exclude filters.
            Components with no remaining parameters after filtering are omitted.

        Example:
            ```python
            # Discover all tunables
            discovery = ComponentDiscovery()
            all_tunables = discovery.discover(component)
            
            # Apply filters
            filtered = discovery._apply_filters(all_tunables)
            
            # With include_patterns=["*.temperature", "*.provider_model"]
            # and exclude_patterns=["*.verbose"], this would keep only
            # temperature and provider_model parameters, excluding verbose ones
            ```

        Note:
            This method creates a new dictionary and does not modify the input.
            If no filters are configured (both include_patterns and exclude_patterns
            are empty), it returns the input dictionary unchanged.
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

    @staticmethod
    def _matches_pattern(path: str, pattern: str) -> bool:
        """
        Check if a path matches a glob pattern. It converts glob patterns to regular
        expressions and performs the matching using Python's `re.match()` function.

        Args:
            path: Dotted path to check (e.g., "classifier_llm.temperature")
            pattern: Glob pattern to match against (e.g., "*.temperature", "classifier_llm.*")

        Returns:
            True if the path matches the pattern, False otherwise.
        """
        # Convert glob pattern to regex
        regex_pattern = pattern.replace('.', r'\.').replace('*', r'.*')
        return re.match(regex_pattern, path) is not None


def discover_tunable_components(component: Any, include_patterns: Optional[List[str]] = None,
                                exclude_patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]:
    """
    This is a simple wrapper around ComponentDiscovery that provides a functional
    interface for discovering tunable components without creating a ComponentDiscovery
    instance. It's used for quick discovery operations or when we don't need
    to reuse the discovery configuration.

    Args:
        component: Root component to search. Can be any Python object containing
                  tunable components in its hierarchy.
        include_patterns: Optional list of glob patterns to include. If provided,
                         only parameters matching at least one pattern will be
                         included in the results.
        exclude_patterns: Optional list of glob patterns to exclude. If provided,
                         parameters matching any pattern will be excluded from
                         the results.

    Returns:
        Dictionary mapping dotted paths to tunable parameter definitions. Same
        structure as ComponentDiscovery.discover().
    """
    discovery = ComponentDiscovery(include_patterns, exclude_patterns)
    return discovery.discover(component)


def build_search_space(discovered: Dict[str, Dict[str, Tuple[ParamType, Any, Any]]]) -> Dict[
    str, Tuple[ParamType, Any, Any]]:
    """
    This method transforms the hierarchical discovery results into a flat
    search space suitable for optimization. It converts the nested structure
    from ComponentDiscovery into a flat dictionary where each parameter is
    identified by its full dotted path.

    The transformation:
    - Input: `{"component": {"param": (type, min, max)}}`
    - Output: `{"component.param": (type, min, max)}`

    This flattened structure is what the optimizer uses to define the search
    space dimensions for parameter optimization.

    Args:
        discovered: Discovery results from ComponentDiscovery.discover() or
                   discover_tunable_components(). Dictionary mapping component
                   paths to their tunable parameters.

    Returns:
        Dictionary mapping full parameter paths to parameter definitions.
        Each key is a dotted path like "classifier_llm.temperature" and each
        value is a tuple containing the parameter type and constraints.
    """
    search_space = {}

    for component_path, tunables in discovered.items():
        for param_name, param_def in tunables.items():
            full_path = f"{component_path}.{param_name}"
            search_space[full_path] = param_def

    return search_space
