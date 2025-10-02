import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import optuna
from optuna.samplers import TPESampler
from ..tunable.types import (
    ParamType, OptimizationMode, Constraints, ScalarizationWeights,
    MetricResult, TrialResult
)

logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization strategies in Octuner.

    This class defines the interface that all optimization strategies must implement
    to work with the LLMOptimizer. It provides a unified way to handle different
    optimization approaches while maintaining compatibility with Optuna.

    Each strategy must implement methods for creating Optuna studies, computing
    objective values from metric results, and determining the best trial from
    completed studies.
    """

    def __init__(self, constraints: Optional[Constraints] = None,
                 scalarization_weights: Optional[ScalarizationWeights] = None):
        self.constraints = constraints or {}
        self.scalarization_weights = scalarization_weights or ScalarizationWeights()

    @abstractmethod
    def create_study(self, study_name: str, seed: Optional[int] = None) -> optuna.Study:
        """
        Create an Optuna study configured for this optimization strategy.
        
        Args:
            study_name: Name for the Optuna study
            seed: Random seed for reproducible optimization
            
        Returns:
            Configured Optuna study ready for optimization
        """
        pass

    @abstractmethod
    def compute_objectives(self, result: MetricResult) -> Tuple[float, ...]:
        """
        Convert the raw metric results (quality, cost, latency) into objective values
        that Optuna can optimize. The conversion depends on the specific strategy (e.g.,
        Pareto uses multiple objectives, scalarized combines them into one).
        
        Args:
            result: MetricResult containing quality, cost, and latency
            
        Returns:
            Tuple of objective values for Optuna optimization
        """
        pass

    @abstractmethod
    def get_fallback_objectives(self) -> Tuple[float, ...]:
        """
        When a trial fails (e.g., due to parameter constraints or errors), this method
        provides objective values that represent the worst possible performance, ensuring
        failed trials are properly ranked.
        
        Returns:
            Tuple of fallback objective values
        """
        pass

    @abstractmethod
    def get_best_trial_from_study(self, study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
        """
        Get the best trial from the study according to this strategy.
        
        Different strategies define "best" differently:
        - Pareto: Highest quality among Pareto-optimal solutions
        - Constrained: Highest quality among constraint-satisfying trials
        - Scalarized: Lowest combined objective score
        
        Args:
            study: Completed Optuna study
            
        Returns:
            Best trial according to this strategy, or None if no trials completed
        """
        pass


class ParetoOptimizationStrategy(OptimizationStrategy):
    """
    Multi-objective Pareto optimization strategy.

    This strategy simultaneously optimizes quality, cost, and latency as separate
    objectives, finding Pareto-optimal solutions that represent the best trade-offs
    between these competing goals. It uses Optuna multi-objective optimization
    capabilities to explore the Pareto frontier.

    Key characteristics:
    - Optimizes three objectives: maximize quality, minimize cost, minimize latency
    - Finds multiple Pareto-optimal solutions representing different trade-offs
    - Best trial is selected as the one with the highest quality among Pareto-optimal solutions
    - Suitable when you want to explore the full range of quality/cost/latency trade-offs
    """

    def create_study(self, study_name: str, seed: Optional[int] = None) -> optuna.Study:
        sampler = TPESampler(seed=seed)
        return optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            directions=["minimize", "minimize", "minimize"]  # 1-quality, cost, latency
        )

    def compute_objectives(self, result: MetricResult) -> Tuple[float, ...]:
        objectives = [1.0 - result.quality]

        # Cost (minimize, with fallback to 0)
        cost_obj = result.cost if result.cost is not None else 0.0
        objectives.append(cost_obj)

        # Latency (minimize, with fallback to 0)
        latency_obj = result.latency_ms if result.latency_ms is not None else 0.0
        objectives.append(latency_obj)

        return tuple(objectives)

    def get_fallback_objectives(self) -> Tuple[float, ...]:
        return 1.0, float('inf'), float('inf')

    def get_best_trial_from_study(self, study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
        # For Pareto, find the trial with highest quality
        best_trial = None
        best_quality = -1.0

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                quality = trial.user_attrs.get("quality", 0.0)
                if quality > best_quality:
                    best_quality = quality
                    best_trial = trial

        return best_trial


class ConstrainedOptimizationStrategy(OptimizationStrategy):
    """
    Constrained single-objective optimization strategy.

    This strategy maximizes quality while enforcing hard constraints on cost and
    latency. Trials that violate constraints are pruned (terminated early) to
    avoid wasting computational resources on infeasible solutions.

    Key characteristics:
    - Single objective: maximize quality
    - Hard constraints on cost and/or latency (configurable)
    - Prunes trials that violate constraints using Optuna pruning mechanism
    - Best trial is the one with the highest quality among constraint-satisfying trials
    - Suitable when you have strict deployment requirements (e.g., budget limits)
    """

    def create_study(self, study_name: str, seed: Optional[int] = None) -> optuna.Study:
        sampler = TPESampler(seed=seed)
        return optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            direction="maximize"  # Maximize quality
        )

    def compute_objectives(self, result: MetricResult) -> Tuple[float, ...]:
        # Check constraints
        if result.cost is not None and "cost_total" in self.constraints:
            if result.cost > self.constraints["cost_total"]:
                raise optuna.exceptions.TrialPruned("Cost constraint violated")

        if result.latency_ms is not None and "latency_ms" in self.constraints:
            if result.latency_ms > self.constraints["latency_ms"]:
                raise optuna.exceptions.TrialPruned("Latency constraint violated")

        return (result.quality,)

    def get_fallback_objectives(self) -> Tuple[float, ...]:
        return (0.0,)

    def get_best_trial_from_study(self, study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
        return study.best_trial


class ScalarizedOptimizationStrategy(OptimizationStrategy):
    """
    Scalarized multi-objective optimization strategy.

    This strategy combines quality, cost, and latency into a single objective
    function using configurable weights. It converts the multi-objective problem
    into a single-objective one that can be optimized with standard methods.

    Key characteristics:
    - Single objective: minimize weighted combination of (1-quality), cost, and latency
    - Configurable weights for each component via scalarization_weights
    - Latency is normalized to seconds for consistent scaling
    - Best trial is the one with lowest combined objective score
    - Suitable when you have clear preferences for the relative importance of objectives
    """

    def create_study(self, study_name: str, seed: Optional[int] = None) -> optuna.Study:
        sampler = TPESampler(seed=seed)
        return optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            direction="minimize"  # Minimize combined score
        )

    def compute_objectives(self, result: MetricResult) -> Tuple[float, ...]:
        score = 1.0 - result.quality  # Quality component

        # Add cost component
        if result.cost is not None:
            score += self.scalarization_weights.cost_weight * result.cost

        # Add latency component
        if result.latency_ms is not None:
            score += self.scalarization_weights.latency_weight * (
                    result.latency_ms / 1000.0)  # Convert to seconds

        return (score,)

    def get_fallback_objectives(self) -> Tuple[float, ...]:
        return (float('inf'),)

    def get_best_trial_from_study(self, study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
        return study.best_trial


# Currently supported optimization strategies
OPTIMIZATION_STRATEGIES = {
    "pareto": ParetoOptimizationStrategy,
    "constrained": ConstrainedOptimizationStrategy,
    "scalarized": ScalarizedOptimizationStrategy,
}


def create_optimization_strategy(mode: OptimizationMode, constraints: Optional[Constraints] = None,
                                 scalarization_weights: Optional[ScalarizationWeights] = None) -> OptimizationStrategy:
    """
    Factory to create optimization strategies. It provides a unified interface for
    creating optimization strategies based on the specified mode. It handles the
    configuration of constraints and scalarization weights as needed for each strategy type.

    Args:
        mode: Optimization mode determining the strategy type:
              - "pareto": Multi-objective Pareto optimization
              - "constrained": Single-objective with hard constraints
              - "scalarized": Multi-objective converted to single-objective
        constraints: Hard constraints for constrained mode. Dictionary with
                    constraint names and maximum values (e.g., {"cost_total": 0.01})
        scalarization_weights: Weights for scalarized mode. Defines relative
                              importance of quality, cost, and latency objectives
        
    Returns:
        Configured optimization strategy instance
        
    Raises:
        ValueError: If an unknown optimization mode is specified
    """
    if mode not in OPTIMIZATION_STRATEGIES:
        raise ValueError(
            f"Unknown optimization mode: {mode}. "
            f"Available modes: {list(OPTIMIZATION_STRATEGIES.keys())}"
        )

    strategy_class = OPTIMIZATION_STRATEGIES[mode]
    return strategy_class(constraints, scalarization_weights)


class LLMOptimizer:
    """
    Core optimization engine that uses Optuna to find optimal parameter configurations.

    LLMOptimizer is the main optimization engine that coordinates the search for
    optimal parameter values using Optuna's sophisticated optimization algorithms.
    It integrates with DatasetExecutor to evaluate parameter configurations and
    uses optimization strategies to determine the best solutions.

    The optimization process:
    1. **Parameter Suggestion**: Uses Optuna to suggest parameter values from the search space
    2. **Trial Execution**: Evaluates suggested parameters using DatasetExecutor
    3. **Objective Computation**: Converts evaluation results to objective values using the strategy
    4. **Optimization**: Uses Optuna's TPE sampler to intelligently explore the parameter space
    5. **Result Analysis**: Extracts best parameters and trial results from completed optimization

    Key features:
    - Supports multiple optimization strategies (Pareto, constrained, scalarized)
    - Intelligent parameter suggestion using TPE (Tree-structured Parzen Estimator)
    - Robust error handling with fallback objective values for failed trials
    - Comprehensive trial result tracking and analysis
    - Integration with Optuna's pruning mechanism for constraint handling
    """

    def __init__(self, search_space: Dict[str, Tuple[ParamType, Any, Any]], mode: OptimizationMode = "pareto",
                 constraints: Optional[Constraints] = None,
                 scalarization_weights: Optional[ScalarizationWeights] = None,
                 seed: Optional[int] = None):
        """
        Initialize the LLMOptimizer with search space and optimization configuration.

        Args:
            search_space: Dictionary mapping parameter paths to their definitions.
                        Each definition is a tuple of (type, min_value, max_value)
                        for numeric parameters or (type, choices) for categorical parameters.
            mode: Optimization strategy to use:
                 - "pareto": Multi-objective Pareto optimization (default)
                 - "constrained": Single-objective with hard constraints
                 - "scalarized": Multi-objective converted to single-objective
            constraints: Hard constraints for constrained mode. Dictionary with
                        constraint names and maximum values (e.g., {"cost_total": 0.01}).
            scalarization_weights: Weights for scalarized mode. Defines relative
                                 importance of quality, cost, and latency objectives.
            seed: Random seed for reproducible optimization results. Use the same
                 seed to get identical optimization runs.
        """
        self.search_space = search_space
        self.mode = mode
        self.seed = seed

        # Create optimization strategy
        self.strategy = create_optimization_strategy(mode, constraints, scalarization_weights)

        # Create Optuna study
        self.study = self._create_study()

    def _create_study(self) -> optuna.Study:
        """
        Create an Optuna study using the optimization strategy.
        
        Returns:
            Configured Optuna study
        """
        study_name = f"octuner_{self.mode}"
        return self.strategy.create_study(study_name, self.seed)

    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameter values for an Optuna trial.

        This method uses Optuna's intelligent parameter suggestion to generate
        parameter values from the search space. It handles different parameter
        types (float, int, choice, bool, list) and converts them to appropriate
        Optuna suggestion calls.

        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Dictionary mapping parameter paths to suggested values
            
        Raises:
            TypeError: If parameter ranges are invalid for numeric types
            ValueError: If an unknown parameter type is encountered
        """
        parameters = {}

        for param_path, param_def in self.search_space.items():
            param_type = param_def[0]
            if param_type == "float":
                try:
                    parameters[param_path] = trial.suggest_float(param_path, param_def[1], param_def[2])
                except TypeError as e:
                    raise TypeError(f"Invalid float range for {param_path}: {param_def[1]} to {param_def[2]}") from e
            elif param_type == "int":
                parameters[param_path] = trial.suggest_int(param_path, param_def[1], param_def[2])
            elif param_type == "choice":
                parameters[param_path] = trial.suggest_categorical(param_path, param_def[1])
            elif param_type == "bool":
                parameters[param_path] = trial.suggest_categorical(param_path, [True, False])
            elif param_type == "list":
                # For list types, convert to string representations for Optuna
                import json
                str_choices = [json.dumps(choice) if choice != "" else "" for choice in param_def[1]]
                parameters[param_path] = trial.suggest_categorical(param_path, str_choices)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return parameters

    def objective_function(self, trial: optuna.Trial, executor: Any, replicates: int = 1) -> Tuple[float, ...]:
        """
        Objective function for Optuna optimization.

        This method serves as the objective function that Optuna optimizes. It
        suggests parameters, executes the trial using the DatasetExecutor, and
        converts the results to objective values using the optimization strategy.

        The function handles errors gracefully by returning fallback objective
        values for failed trials, ensuring the optimization process continues
        even when individual trials fail.

        Args:
            trial: Optuna trial object for parameter suggestion
            executor: DatasetExecutor instance for trial evaluation
            replicates: Number of replicates to run for statistical robustness
            
        Returns:
            Tuple of objective values for Optuna optimization
        """
        # Suggest parameters
        parameters = self.suggest_parameters(trial)

        # Execute trial
        try:
            result = executor.execute_with_replicates(parameters, replicates)

            # Store trial info
            trial.set_user_attr("quality", result.quality)
            trial.set_user_attr("cost", result.cost)
            trial.set_user_attr("latency_ms", result.latency_ms)

            # Use strategy to compute objectives
            return self.strategy.compute_objectives(result)

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            # Return worst possible values using strategy
            return self.strategy.get_fallback_objectives()

    def optimize(self, executor: Any, max_trials: int = 120, replicates: int = 1,
                 timeout: Optional[float] = None) -> List[TrialResult]:
        """
        Run the optimization process to find optimal parameter configurations.

        This method orchestrates the entire optimization process using Optuna.
        It runs the specified number of trials, each evaluating a different
        parameter configuration, and returns comprehensive results from all trials.

        Args:
            executor: DatasetExecutor instance for evaluating parameter configurations
            max_trials: Maximum number of optimization trials to run. More trials
                       generally lead to better results but take longer.
            replicates: Number of replicates per trial for statistical robustness.
                       Higher values reduce noise but increase computation time.
            timeout: Maximum time in seconds for the entire optimization process.
                    If None, optimization runs until max_trials is reached.
            
        Returns:
            List of TrialResult objects containing results from all completed trials,
            including both successful and failed trials with error information.
        """
        logger.info(f"Starting optimization with mode: {self.mode}")
        logger.info(f"Search space: {len(self.search_space)} parameters")

        # Run optimization
        self.study.optimize(
            lambda trial: self.objective_function(trial, executor, replicates),
            n_trials=max_trials,
            timeout=timeout
        )

        # Convert results to our format
        trial_results = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = TrialResult(
                    trial_number=trial.number,
                    parameters=trial.params,
                    metrics=MetricResult(
                        quality=trial.user_attrs.get("quality", 0.0),
                        cost=trial.user_attrs.get("cost"),
                        latency_ms=trial.user_attrs.get("latency_ms")
                    ),
                    success=True
                )
            else:
                result = TrialResult(
                    trial_number=trial.number,
                    parameters=trial.params,
                    metrics=MetricResult(quality=0.0),
                    success=False,
                    error=f"Trial state: {trial.state}"
                )

            trial_results.append(result)

        logger.info(f"Optimization completed: {len(trial_results)} trials")
        return trial_results

    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best parameter configuration found during optimization.

        This method returns the parameter values from the best trial according
        to the optimization strategy. The definition of "best" depends on the
        strategy used (e.g., highest quality for Pareto, lowest combined score for scalarized).

        Returns:
            Dictionary mapping parameter paths to their optimal values.
            Returns empty dictionary if no trials completed successfully.
        """
        best_trial = self.strategy.get_best_trial_from_study(self.study)
        return best_trial.params if best_trial else {}

    def get_best_trial(self) -> Optional[TrialResult]:
        """
        Get the best trial result from the optimization.

        This method returns the complete TrialResult object for the best trial,
        including parameters, metrics, and success status. The best trial is
        determined by the optimization strategy used.

        Returns:
            TrialResult object for the best trial, or None if no trials
            completed successfully.
        """
        if not self.study.trials:
            return None

        best_trial = self.strategy.get_best_trial_from_study(self.study)

        if best_trial:
            return TrialResult(
                trial_number=best_trial.number,
                parameters=best_trial.params,
                metrics=MetricResult(
                    quality=best_trial.user_attrs.get("quality", 0.0),
                    cost=best_trial.user_attrs.get("cost"),
                    latency_ms=best_trial.user_attrs.get("latency_ms")
                ),
                success=True
            )

        return None
