"""
Core type definitions for Octuner.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass

# Parameter types supported by the tuner
ParamType = Literal["float", "int", "choice", "bool"]

# Optimization modes
OptimizationMode = Literal["pareto", "constrained", "scalarized"]

# Metric result structure
@dataclass
class MetricResult:
    """
    Result of a single metric evaluation.
    """
    quality: float  # Required: [0, 1] score
    cost: Optional[float] = None  # Optional: cost in user currency
    latency_ms: Optional[float] = None  # Optional: latency in milliseconds

# Trial result structure
@dataclass
class TrialResult:
    """
    Result of a single optimization trial.
    """
    trial_number: int
    parameters: Dict[str, Any]
    metrics: MetricResult
    success: bool = True
    error: Optional[str] = None

# Search result structure
@dataclass
class SearchResult:
    """
    Result of an optimization search.
    """
    best_trial: TrialResult
    all_trials: List[TrialResult]
    optimization_mode: OptimizationMode
    dataset_size: int
    total_trials: int
    best_parameters: Dict[str, Any]
    metrics_summary: Dict[str, float]
    
    def save_best(self, path: str) -> None:
        """
        Save best parameters to YAML file.
        """
        from ..utils.exporter import save_parameters_to_yaml
        save_parameters_to_yaml(self.best_parameters, path, self.metrics_summary)

# Dataset structure
DatasetItem = Dict[str, Any]  # {"input": X, "target": Y}
Dataset = List[DatasetItem]

# Metric function type
MetricFunction = callable  # (output: Any, target: Any) -> float

# Entrypoint function type
EntrypointFunction = callable  # (component: Any, input: Any) -> Any

# Constraint specification
Constraints = Dict[str, Union[float, int]]  # e.g., {"latency_ms": 1000, "cost_total": 0.01}

# Scalarization weights
@dataclass
class ScalarizationWeights:
    """
    Weights for scalarized optimization mode.
    """
    cost_weight: float = 1.0
    latency_weight: float = 1.0
