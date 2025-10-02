"""
Component discovery for Octuner.

This module contains functionality for discovering tunable components in the component tree.
It provides the ComponentDiscovery class and convenience functions that automatically
find and catalog all tunable parameters within complex object hierarchies, building
the search space needed for optimization.

Key components:
- ComponentDiscovery: Main class for discovering tunable components
- discover_tunable_components(): Convenience function for quick discovery
- build_search_space(): Converts discovery results to optimization search space

The discovery process is a critical first step in the optimization workflow,
automatically identifying which parameters can be tuned and their constraints.
"""

from .discovery import ComponentDiscovery, build_search_space, discover_tunable_components

__all__ = [
    "ComponentDiscovery",
    "build_search_space",
    "discover_tunable_components",
]
