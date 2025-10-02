"""
Optimization algorithms and execution for Octuner.

This module contains the core optimization infrastructure for automatically tuning
LLM components and complex AI systems. It provides sophisticated optimization
algorithms, parallel execution capabilities, and the main AutoTuner orchestrator.

Key components:
- AutoTuner: Main orchestrator for parameter optimization workflows
- LLMOptimizer: Core optimization engine with multiple strategies
- DatasetExecutor: Parallel execution of optimization trials
- Optimization strategies: Pareto, constrained, and scalarized optimization

The optimization process automatically discovers tunable parameters, builds
multi-dimensional search spaces, and uses intelligent algorithms to find
optimal configurations that balance quality, cost, and latency objectives.
"""

from .auto import AutoTuner
from .optimizer import LLMOptimizer, create_optimization_strategy, OPTIMIZATION_STRATEGIES
from .executor import DatasetExecutor, execute_trial

__all__ = [
    "AutoTuner",
    "LLMOptimizer", 
    "create_optimization_strategy",
    "OPTIMIZATION_STRATEGIES",
    "DatasetExecutor",
    "execute_trial",
]
