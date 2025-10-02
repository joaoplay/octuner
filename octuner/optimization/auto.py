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
    This is the main orchestrator for auto-tuning the LLM components. As the central
    class of the Octuner optimization system, it that coordinates the entire parameter
    optimization workflow. It automatically discovers tunable parameters in complex
    component hierarchies, builds search spaces, and runs optimization algorithms to
    find the best configuration.

    How it works:

    1. **Component Discovery**: Automatically finds all tunable components in the
       object hierarchy using ComponentDiscovery, identifying parameters that can
       be optimized (temperature, model selection, provider choices, etc.)

    2. **Search Space Construction**: Builds a multi-dimensional search space from
       discovered parameters, defining the optimization landscape with parameter
       types, ranges, and constraints.

    3. **Optimization Execution**: Runs intelligent search algorithms (Pareto,
       constrained, or scalarized optimization) to explore the search space and
       find optimal parameter combinations.

    4. **Result Analysis**: Provides comprehensive results including the best parameters,
       performance metrics, and detailed trial information for analysis and application.

    Key Features:

    - **Multi-Objective Optimization**: Supports Pareto optimization for balancing
      quality, cost, and latency objectives
    - **Constraint Handling**: Supports hard constraints for real-world deployment
      requirements
    - **Scalarization**: Converts multi-objective problems to single-objective
      optimization with custom weights
    - **Parallel Execution**: Supports concurrent trial execution for faster optimization
    - **Flexible Filtering**: Include/exclude specific parameters to focus optimization
    - **Reproducible Results**: Seed support for consistent optimization runs

    Example:
        ```python
        from octuner import AutoTuner, MultiProviderTunableLLM
        
        # Create a tunable component
        llm = MultiProviderTunableLLM(config_file="config.yaml")
        
        # Define evaluation function and dataset
        def evaluate(component, input_data):
            result = component.call(input_data["text"])
            return {"quality": compute_quality(result, input_data["target"])}
        
        dataset = [{"text": "Hello", "target": "Hi there"}]
        
        # Create and configure tuner
        tuner = AutoTuner(
            component=llm,
            entrypoint=evaluate,
            dataset=dataset,
            metric=lambda output, target: output["quality"]
        )
        
        # Focus on specific parameters
        tuner.include(["*.temperature", "*.provider_model"])
        
        # Run optimization
        result = tuner.search(max_trials=50, mode="pareto")
        
        # Apply the best parameters
        from octuner import apply_best
        apply_best(llm, result.best_parameters)
        ```

    Optimization Modes:

        - **"pareto"**: Multi-objective optimization finding Pareto-optimal solutions
        - **"constrained"**: Single-objective optimization with hard constraints
        - **"scalarized"**: Multi-objective converted to single-objective with weights
    """

    def __init__(self, component: Any, entrypoint: EntrypointFunction = None, dataset: Dataset = None,
                 metric: MetricFunction = None, entrypoint_function: EntrypointFunction = None,
                 metric_function: MetricFunction = None, max_workers: int = 1,
                 optimization_mode: str = "pareto", n_trials: int = 120,
                 constraints: Optional[Constraints] = None,
                 scalarization_weights: Optional[ScalarizationWeights] = None):
        """
        Initialize the AutoTuner with a component and evaluation setup.

        This constructor sets up the AutoTuner with all necessary components for
        parameter optimization. It validates the inputs and prepares the internal
        state for discovery and optimization.
        
        Args:
            component: The component to optimize. Must contain tunable parameters
                      (implement TunableMixin or be registered as tunable). Can be
                      a single component or a complex hierarchy of components.
            entrypoint: Function that evaluates the component with input data.
                       Called as `entrypoint(component, input_data)` for each
                       dataset item. Should return a dictionary with metrics.
                       (Legacy parameter name)
            dataset: List of input/target pairs for evaluation. Each item should
                    contain the input data and expected output for evaluation.
            metric: Function that computes quality scores from evaluation results.
                   Called as `metric(output, target)` where output is the result
                   from entrypoint and target is the expected output.
                   (Legacy parameter name)
            entrypoint_function: Same as entrypoint but with clearer naming.
            metric_function: Same as metric but with clearer naming.
            max_workers: Maximum number of concurrent workers for parallel
                        evaluation during optimization trials. Higher values
                        speed up optimization but use more resources.
            optimization_mode: Optimization strategy to use. Options:
                             - "pareto": Multi-objective optimization (default)
                             - "constrained": Single-objective with constraints
                             - "scalarized": Multi-objective with custom weights
            n_trials: Default number of optimization trials to run. Can be
                     overridden in search() calls.
            constraints: Hard constraints for constrained optimization mode.
                        Dictionary with constraint names and values.
            scalarization_weights: Weights for scalarized optimization mode.
                                 Dictionary mapping objective names to weights.
        """
        self.component = component

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
        Create an AutoTuner instance using the factory pattern. It's the recommended way
        to create AutoTuner instances for most use cases.

        Args:
            component: The component to optimize. Must contain tunable parameters
                      (implement TunableMixin or be registered as tunable).
            entrypoint: Function that evaluates the component with input data.
                       Called as `entrypoint(component, input_data)` for each
                       dataset item. Should return a dictionary with metrics.
            dataset: List of input/target pairs for evaluation. Each item should
                    contain the input data and expected output for evaluation.
            metric: Function that computes quality scores from evaluation results.
                   Called as `metric(output, target)` where output is the result
                   from entrypoint and target is the expected output.
            max_workers: Maximum number of concurrent workers for parallel
                        evaluation during optimization trials. Default is 1.

        Returns:
            Configured AutoTuner instance ready for optimization.

        Example:
            ```python
            # Create tuner using factory method
            tuner = AutoTuner.from_component(
                component=my_llm,
                entrypoint=lambda comp, data: comp.call(data["text"]),
                dataset=test_dataset,
                metric=lambda output, target: compute_quality(output, target),
                max_workers=4
            )
            
            # Run optimization
            result = tuner.search(max_trials=100)
            ```
        """
        return cls(component, entrypoint, dataset, metric, max_workers)

    def include(self, patterns: List[str]) -> "AutoTuner":
        """
        This method allows to narrow down the optimization to only specific parameters
        or components, reducing the search space and focusing on the most important
        parameters for your use case.

        Args:
            patterns: List of glob patterns to include in optimization.
                     Only parameters matching at least one pattern will be
                     included in the search space. Examples:
                     - ["*.temperature"]: Include all temperature parameters
                     - ["*.provider_model"]: Include all provider/model selections
                     - ["classifier_llm.*"]: Include all parameters in classifier_llm
                     - ["*.max_tokens", "*.top_p"]: Include max_tokens and top_p

        Returns:
            Self for method chaining, allowing fluent interface.

        Example:
            ```python
            # Focus on core LLM parameters
            tuner.include(["*.temperature", "*.max_tokens", "*.provider_model"])
            
            # Focus on specific components
            tuner.include(["classifier_llm.*", "confidence_llm.*"])
            
            # Chain with exclude for fine control
            tuner.include(["*.temperature"]).exclude(["*.verbose"])
            ```

        Note:
            Include patterns are applied before exclude patterns. If no include
            patterns are set, all discovered parameters are considered for inclusion.
        """
        self.discovery.include_patterns = patterns
        return self

    def exclude(self, patterns: List[str]) -> "AutoTuner":
        """
        This method allows to exclude certain parameters from the optimization process,
        typically to remove debug parameters, verbose settings, or other parameters that
        shouldn't be optimized.

        Args:
            patterns: List of glob patterns to exclude from optimization.
                     Parameters matching any pattern will be removed from
                     the search space. Examples:
                     - ["*.verbose"]: Exclude all verbose parameters
                     - ["*.debug", "*.log_level"]: Exclude debug and logging parameters
                     - ["*.frequency_penalty"]: Exclude specific parameters
                     - ["nested.component.*"]: Exclude all parameters in nested.component

        Returns:
            Self for method chaining

        Example:
            ```python
            # Exclude debug parameters
            tuner.exclude(["*.verbose", "*.debug", "*.log_level"])
            
            # Exclude less important parameters
            tuner.exclude(["*.frequency_penalty", "*.presence_penalty"])
            
            # Chain with include for precise control
            tuner.include(["*.temperature"]).exclude(["*.verbose"])
            ```

        Note:
            Exclude patterns are applied after include patterns. If a parameter
            matches both include and exclude patterns, it will be excluded.
        """
        self.discovery.exclude_patterns = patterns
        return self

    def build_search_space(self) -> None:
        """
        Performs the component discovery process and constructs the search space that
        defines the optimization landscape. It's automatically called by search() if
        not already built, but can be called manually to inspect the discovered
        parameters.

        The discovery process:

        1. **Component Traversal**: Recursively explores the component hierarchy
        2. **Parameter Detection**: Identifies all tunable parameters using
           TunableMixin protocol or registry-based detection
        3. **Search Space Construction**: Builds a flat dictionary mapping
           parameter paths to their definitions and constraints
        4. **Validation**: Ensures at least one tunable parameter is found

        The resulting search space contains:

        - Parameter paths (e.g., "llm.temperature", "classifier.max_tokens")
        - Parameter types ("float", "int", "choice", "bool")
        - Value ranges or choices for each parameter
        - Default values where applicable

        Returns:
            None. The search space is stored in self.search_space.

        Raises:
            ValueError: If no tunable components are found in the component
                       hierarchy. This usually means the component doesn't
                       implement TunableMixin or isn't registered as tunable.

        Example:
            ```python
            # Build search space manually
            tuner.build_search_space()

            # Inspect discovered parameters
            summary = tuner.get_search_space_summary()
            print(f"Found {summary['total_parameters']} tunable parameters")
            print(f"Parameter types: {summary['parameter_types']}")

            # View specific parameters
            for param_path, param_def in tuner.search_space.items():
                print(f"{param_path}: {param_def}")
            ```

        Note:
            This method is idempotent - calling it multiple times has no effect
            after the first successful call. The search space is cached until
            the component structure changes.
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
        Run the optimization search to find the best parameter configuration.

        This is the main method that orchestrates the entire optimization process. It
        automatically discovers tunable parameters, sets up the optimization environment,
        and runs intelligent search algorithms to find optimal parameter combinations.

        The optimization process:

        1. **Discovery**: Finds all tunable parameters in the component hierarchy
        2. **Search Space Setup**: Builds the multi-dimensional search space
        3. **Optimization**: Runs the specified optimization algorithm
        4. **Result Analysis**: Analyzes results and returns comprehensive findings

        Args:
            max_trials: Maximum number of optimization trials to run. More trials
                       generally lead to better results but take longer. Typical
                       values range from 50-500 depending on search space size.
            mode: Optimization strategy to use:
                 - "pareto": Multi-objective optimization finding Pareto-optimal
                   solutions that balance multiple objectives (quality, cost, latency)
                 - "constrained": Single-objective optimization with hard constraints
                   for real-world deployment requirements
                 - "scalarized": Multi-objective converted to single-objective
                   using custom weights for different objectives
            constraints: Hard constraints for constrained optimization mode.
                        Dictionary with constraint names and maximum values.
                        Example: {"max_cost": 0.01, "max_latency_ms": 1000}
            scalarization_weights: Weights for scalarized optimization mode.
                                 Dictionary mapping objective names to weights.
                                 Weights should sum to 1.0 for best results.
                                 Example: {"quality": 0.7, "cost": 0.3}
            replicates: Number of replicates per trial for statistical robustness.
                       Higher values reduce noise but increase computation time.
                       Default is 1 for faster optimization.
            timeout: Maximum time in seconds for the entire optimization process.
                    If None, optimization runs until max_trials is reached.
            seed: Random seed for reproducible optimization results. Use the same
                 seed to get identical results across runs.

        Returns:
            SearchResult containing:

            - best_trial: The best performing trial with optimal parameters
            - all_trials: List of all trials for detailed analysis
            - best_parameters: Dictionary of best parameter values
            - optimization_mode: The mode used for optimization
            - metrics_summary: Statistical summary of all trials

        Example:
            ```python
            # Basic optimization
            result = tuner.search()
            
            # Advanced optimization with constraints
            result = tuner.search(
                max_trials=200,
                mode="constrained",
                constraints={"max_cost": 0.01, "max_latency_ms": 500},
                replicates=3,
                timeout=3600,  # 1 hour timeout
                seed=42
            )
            
            # Multi-objective optimization with custom weights
            result = tuner.search(
                mode="scalarized",
                scalarization_weights={"quality": 0.8, "cost": 0.2},
                max_trials=100
            )
            
            # Access results
            print(f"Best quality: {result.best_trial.metrics.quality}")
            print(f"Best parameters: {result.best_parameters}")
            ```

        Note:
            The first call to search() will automatically discover tunable components and
            build the search space. Subsequent calls reuse the existing search space
            unless the component structure changes.
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

    def get_search_space_summary(self) -> Dict[str, Any]:
        """
        Provides detailed information about the tunable parameters discovered in the
        component hierarchy, including counts, types, and component distribution. Useful
        for understanding the optimization landscape before running optimization.

        Returns:
            Dictionary containing search space summary with keys: "total_parameters", "parameter_types", "components"

        Example:
            ```python
            # Build search space and get summary
            tuner.build_search_space()
            summary = tuner.get_search_space_summary()
            
            print(f"Total parameters: {summary['total_parameters']}")
            print(f"Parameter types: {summary['parameter_types']}")
            print(f"Components: {summary['components']}")
            
            # Output might be:
            # Total parameters: 12
            # Parameter types: {'float': 6, 'choice': 4, 'int': 2}
            # Components: {'llm': 8, 'classifier_llm': 4}
            ```

        Note:
            This method requires the search space to be built first. It's
            automatically called by search() if not already built.
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
