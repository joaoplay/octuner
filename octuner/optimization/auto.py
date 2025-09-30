import logging
from typing import Any, Dict, List, Optional
from ..discovery import ComponentDiscovery, build_search_space
from .executor import DatasetExecutor
from .optimizer import LLMOptimizer
from ..utils.exporter import create_metadata_summary, compute_dataset_fingerprint
from ..tunable.types import (
    Dataset, MetricFunction, EntrypointFunction, OptimizationMode,
    Constraints, ScalarizationWeights, SearchResult, TrialResult
)

logger = logging.getLogger(__name__)


class AutoTuner:
    """
    Main class for auto-tuning LLM components. This class is responsible for orchestrating the entire optimization
    process: discovery, execution, and optimization.
    
    This class orchestrates the entire optimization process:
    1. Discovers tunable components in the component tree
    2. Builds the search space from discovered parameters
    3. Runs optimization trials using Optuna
    4. Returns results that can be saved and applied
    """

    def __init__(self, component: Any, entrypoint: EntrypointFunction = None, dataset: Dataset = None, 
                 metric: MetricFunction = None, entrypoint_function: EntrypointFunction = None,
                 metric_function: MetricFunction = None, max_workers: int = 1, 
                 optimization_mode: str = "pareto", n_trials: int = 120, 
                 constraints: Optional[Constraints] = None, scalarization_weights: Optional[ScalarizationWeights] = None, **kwargs):
        """
        Initialize the AutoTuner.
        
        Args:
            component: Component to tune
            entrypoint: Function to call with (component, input) (legacy parameter)
            dataset: Dataset of input/target pairs
            metric: Function to compute quality score (legacy parameter)
            entrypoint_function: Function to call with (component, input) (new parameter)
            metric_function: Function to compute quality score (new parameter)
            max_workers: Maximum number of concurrent workers for dataset execution
            **kwargs: Additional parameters for backward compatibility
        """
        self.component = component
        
        # Handle both old and new parameter names for backward compatibility
        self.entrypoint = entrypoint or entrypoint_function
        self.dataset = dataset
        self.metric = metric or metric_function
        self.max_workers = max_workers
        self.optimization_mode = optimization_mode
        self.n_trials = n_trials
        self.constraints = constraints
        self.scalarization_weights = scalarization_weights
        
        # Validate required parameters
        if self.entrypoint is None:
            raise ValueError("entrypoint or entrypoint_function is required")
        if self.dataset is None:
            raise ValueError("dataset is required")
        if self.metric is None:
            raise ValueError("metric or metric_function is required")
        if not self.dataset:
            raise ValueError("dataset cannot be empty")
        
        # Check if component has tunable parameters
        if hasattr(self.component, 'get_tunable_parameters'):
            tunable_params = self.component.get_tunable_parameters()
            if not tunable_params:
                raise ValueError("Component has no tunable parameters")
        
        # Validate optimization mode
        valid_modes = ["pareto", "constrained", "scalarized"]
        if self.optimization_mode not in valid_modes:
            raise ValueError(f"Invalid optimization mode: {self.optimization_mode}. Valid modes: {valid_modes}")
        
        # Validate n_trials
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got: {self.n_trials}")

        # Discovery and search space
        self.discovery = ComponentDiscovery()
        self.search_space: Dict[str, Any] = {}

        # Executor and optimizer
        self.executor: Optional[DatasetExecutor] = None
        self.optimizer: Optional[LLMOptimizer] = None

        # Results
        self.trial_results: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None

    @classmethod
    def from_component(cls, *, component: Any, entrypoint: EntrypointFunction, dataset: Dataset, metric: MetricFunction,
                       max_workers: int = 1) -> "AutoTuner":
        """
        Create an AutoTuner instance from a component.
        
        Args:
            component: Component to tune
            entrypoint: Function to call with (component, input)
            dataset: Dataset of input/target pairs
            metric: Function to compute quality score
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Configured AutoTuner instance
        """
        return cls(component, entrypoint, dataset, metric, max_workers)

    def include(self, patterns: List[str]) -> "AutoTuner":
        """
        Set include patterns for component discovery.
        
        Args:
            patterns: Glob patterns to include (e.g., ["*.temperature"])
            
        Returns:
            Self for method chaining
        """
        self.discovery.include_patterns = patterns
        return self

    def exclude(self, patterns: List[str]) -> "AutoTuner":
        """
        Set exclude patterns for component discovery.
        
        Args:
            patterns: Glob patterns to exclude (e.g., ["*.verbose"])
            
        Returns:
            Self for method chaining
        """
        self.discovery.exclude_patterns = patterns
        return self

    def build_search_space(self) -> None:
        """
        Discover tunable components and build search space.
        
        This method must be called before using any search space dependent methods.
        It's safe to call multiple times - subsequent calls will be ignored.
        """
        if self.search_space:
            logger.info("Search space already built, skipping discovery.")
            return  # Already built

        logger.info("Discovering tunable components...")

        # Discover components
        discovered = self.discovery.discover(self.component)

        if not discovered:
            raise ValueError("No tunable components found. Make sure your LLM classes implement TunableMixin.")

        # Build search space using the utility function from discovery.py
        self.search_space = build_search_space(discovered)

        logger.info(f"Discovered {len(discovered)} components with {len(self.search_space)} tunable parameters")

        # Log discovered parameters
        for param_path, param_def in self.search_space.items():
            param_type = param_def[0]
            if param_type in ["choice", "list", "bool"]:
                logger.debug(f"  {param_path}: {param_type} ({param_def[1]})")
            else:
                logger.debug(f"  {param_path}: {param_type} ({param_def[1]}, {param_def[2]})")

    def _setup_executor(self) -> None:
        """
        Set up the dataset executor.
        """
        if self.executor is None:
            self.executor = DatasetExecutor(self.component, self.entrypoint, self.dataset, self.metric,
                                            self.max_workers)

    def _setup_optimizer(self, mode: OptimizationMode, constraints: Optional[Constraints] = None,
                         scalarization_weights: Optional[ScalarizationWeights] = None,
                         seed: Optional[int] = None) -> None:
        """
        Set up the optimizer.
        
        Args:
            mode: Optimization mode
            constraints: Hard constraints for constrained mode
            scalarization_weights: Weights for scalarized mode
            seed: Random seed for reproducibility
        """
        if self.optimizer is None:
            self.optimizer = LLMOptimizer(
                self.search_space,
                mode=mode,
                constraints=constraints,
                scalarization_weights=scalarization_weights,
                seed=seed
            )

    def search(self, *, max_trials: int = 120, mode: OptimizationMode = "pareto",
               constraints: Optional[Constraints] = None, scalarization_weights: Optional[ScalarizationWeights] = None,
               replicates: int = 1, timeout: Optional[float] = None, seed: Optional[int] = None) -> SearchResult:
        """
        Run the optimization search.
        
        Args:
            max_trials: Maximum number of trials to run
            mode: Optimization mode ("pareto", "constrained", or "scalarized")
            constraints: Hard constraints for constrained mode
            scalarization_weights: Weights for scalarized mode
            replicates: Number of replicates per trial
            timeout: Timeout in seconds
            seed: Random seed for reproducibility
            
        Returns:
            SearchResult with optimization results
            
        Raises:
            ValueError: If no tunable components are found
        """
        # Initialize search space if not done yet
        self.build_search_space()

        # Set up executor and optimizer
        self._setup_executor()
        self._setup_optimizer(mode, constraints, scalarization_weights, seed)

        # Run optimization
        logger.info(f"Starting optimization with {max_trials} trials, mode: {mode}")
        search_result = self.optimizer.optimize(
            self.executor,
            max_trials=max_trials,
            replicates=replicates,
            timeout=timeout
        )
        
        # Extract trial results from search result
        self.trial_results = search_result.all_trials if hasattr(search_result, 'all_trials') else []

        # Get best trial
        self.best_trial = self.optimizer.get_best_trial()

        if not self.best_trial:
            raise RuntimeError("No successful trials completed")

        # Get best parameters
        best_parameters = self.optimizer.get_best_parameters()

        # Create metadata summary
        dataset_fingerprint = compute_dataset_fingerprint(self.dataset)
        metadata = create_metadata_summary(
            trials=self.trial_results,
            optimization_mode=mode,
            dataset_size=len(self.dataset),
            total_trials=len(self.trial_results),
            best_quality=self.best_trial.metrics.quality,
            best_cost=self.best_trial.metrics.cost,
            best_latency_ms=self.best_trial.metrics.latency_ms,
            dataset_fingerprint=dataset_fingerprint
        )

        # Create search result
        result = SearchResult(
            best_trial=self.best_trial,
            all_trials=self.trial_results,
            optimization_mode=mode,
            dataset_size=len(self.dataset),
            total_trials=len(self.trial_results),
            best_parameters=best_parameters,
            metrics_summary=metadata["metrics_summary"]
        )

        quality = getattr(self.best_trial.metrics, 'quality', 0.0)
        try:
            logger.info(f"Optimization completed. Best quality: {quality:.3f}")
        except (TypeError, ValueError):
            logger.info(f"Optimization completed. Best quality: {quality}")

        return result

    def optimize(self, max_trials: int = None, mode: OptimizationMode = None,
                 constraints: Optional[Constraints] = None, scalarization_weights: Optional[ScalarizationWeights] = None,
                 replicates: int = 1, timeout: Optional[float] = None, seed: Optional[int] = None) -> SearchResult:
        """
        Run optimization (alias for search method for backward compatibility).
        
        Args:
            max_trials: Maximum number of trials to run (defaults to self.n_trials)
            mode: Optimization mode ("pareto", "constrained", or "scalarized") (defaults to self.optimization_mode)
            constraints: Hard constraints for constrained mode
            scalarization_weights: Weights for scalarized mode
            replicates: Number of replicates per trial
            timeout: Timeout in seconds
            seed: Random seed for reproducibility
            
        Returns:
            SearchResult with optimization results
        """
        # Use stored parameters as defaults
        if max_trials is None:
            max_trials = self.n_trials
        if mode is None:
            mode = self.optimization_mode
        if constraints is None:
            constraints = self.constraints
        if scalarization_weights is None:
            scalarization_weights = self.scalarization_weights
            
        return self.search(
            max_trials=max_trials,
            mode=mode,
            constraints=constraints,
            scalarization_weights=scalarization_weights,
            replicates=replicates,
            timeout=timeout,
            seed=seed
        )

    def get_search_space_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the discovered search space.
        
        Returns:
            Dictionary with search space information
            
        Raises:
            ValueError: If search space has not been built yet
        """
        if not self.search_space:
            raise ValueError("Search space not built. Call build_search_space() first.")

        summary = {
            "total_parameters": len(self.search_space),
            "parameter_types": {},
            "components": {}
        }

        # Count parameter types
        for param_path, param_def in self.search_space.items():
            param_type = param_def[0]
            if param_type not in summary["parameter_types"]:
                summary["parameter_types"][param_type] = 0
            summary["parameter_types"][param_type] += 1

        # Group by component
        for param_path in self.search_space.keys():
            component_path = param_path.rsplit('.', 1)[0]
            if component_path not in summary["components"]:
                summary["components"][component_path] = 0
            summary["components"][component_path] += 1

        return summary

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameter values on the component.
        
        Returns:
            Dictionary of current parameter values
            
        Raises:
            ValueError: If search space has not been built yet
        """
        if not self.search_space:
            raise ValueError("Search space not built. Call build_search_space() first.")

        from ..utils.setter import get_parameter

        current_params = {}
        for param_path in self.search_space.keys():
            try:
                value = get_parameter(self.component, param_path)
                current_params[param_path] = value
            except Exception as e:
                logger.warning(f"Could not get current value for {param_path}: {e}")

        return current_params

    def _get_parameter_suggestions(self) -> Dict[str, Any]:
        """
        Get parameter suggestions based on discovered tunable parameters.
        
        Returns:
            Dictionary with parameter suggestions
        """
        if not self.search_space:
            self.build_search_space()
        
        suggestions = {}
        for param_path, param_def in self.search_space.items():
            param_type = param_def[0]
            # Strip leading dot from parameter path for cleaner names
            clean_path = param_path.lstrip('.')
            if param_type in ["choice", "list", "bool"]:
                suggestions[clean_path] = {
                    "type": param_type,
                    "options": param_def[1]
                }
            else:
                # param_def is (type, min, max) for numeric types
                if len(param_def) >= 3:
                    suggestions[clean_path] = {
                        "type": param_type,
                        "range": (param_def[1], param_def[2]),
                        "default": param_def[2] if len(param_def) > 3 else None
                    }
                else:
                    suggestions[clean_path] = {
                        "type": param_type,
                        "range": (param_def[1], param_def[2]) if len(param_def) >= 2 else None
                    }
        
        return suggestions
