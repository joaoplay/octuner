import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import optuna
from optuna.samplers import TPESampler
from ..tunable.types import (
    ParamType, OptimizationMode, Constraints, ScalarizationWeights,
    MetricResult, TrialResult
)
from ..discovery import build_search_space

logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization strategies.
    """
    
    def __init__(self, constraints: Optional[Constraints] = None,
                 scalarization_weights: Optional[ScalarizationWeights] = None):
        self.constraints = constraints or {}
        self.scalarization_weights = scalarization_weights or ScalarizationWeights()
    
    @abstractmethod
    def create_study(self, study_name: str, seed: Optional[int] = None) -> optuna.Study:
        """
        Create an Optuna study for this optimization strategy.
        """
        pass
    
    @abstractmethod
    def compute_objectives(self, result: MetricResult) -> Tuple[float, ...]:
        """
        Compute objective values from a metric result.
        """
        pass
    
    @abstractmethod
    def get_fallback_objectives(self) -> Tuple[float, ...]:
        """
        Get fallback objective values for failed trials.
        """
        pass
    
    @abstractmethod
    def get_best_trial_from_study(self, study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
        """
        Get the best trial from the study according to this strategy.
        """
        pass


class ParetoOptimizationStrategy(OptimizationStrategy):
    """
    Multi-objective Pareto optimization strategy.
    Optimizes quality, cost, and latency simultaneously.
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
    Constrained optimization strategy.
    Maximizes quality subject to cost and latency constraints.
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
    Scalarized optimization strategy.
    Combines quality, cost, and latency into a single objective.
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


def create_optimization_strategy(mode: OptimizationMode,
                               constraints: Optional[Constraints] = None,
                               scalarization_weights: Optional[ScalarizationWeights] = None) -> OptimizationStrategy:
    """
    Factory function to create optimization strategies.
    
    Args:
        mode: Optimization mode
        constraints: Hard constraints for constrained mode
        scalarization_weights: Weights for scalarized mode
        
    Returns:
        Optimization strategy instance
    """
    if mode == "pareto":
        return ParetoOptimizationStrategy(constraints, scalarization_weights)
    elif mode == "constrained":
        return ConstrainedOptimizationStrategy(constraints, scalarization_weights)
    elif mode == "scalarized":
        return ScalarizedOptimizationStrategy(constraints, scalarization_weights)
    else:
        raise ValueError(f"Unknown optimization mode: {mode}")


class LLMOptimizer:
    """
    Optimizes LLM parameters using Optuna.
    """

    def __init__(self, search_space: Dict[str, Tuple[ParamType, Any, Any]], mode: OptimizationMode = "pareto",
                 constraints: Optional[Constraints] = None,
                 scalarization_weights: Optional[ScalarizationWeights] = None,
                 seed: Optional[int] = None):
        """
        Initialize the optimizer.
         
        Args:
            search_space: Dictionary mapping parameter paths to definitions
            mode: Optimization mode ("pareto", "constrained", or "scalarized")
            constraints: Hard constraints for constrained mode
            scalarization_weights: Weights for scalarized mode
            seed: Random seed for reproducibility
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
        Suggest parameter values for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of parameter values
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
        
        Args:
            trial: Optuna trial object
            executor: DatasetExecutor instance
            replicates: Number of replicates to run
            
        Returns:
            Tuple of objective values
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
        Run the optimization.
        
        Args:
            executor: DatasetExecutor instance
            max_trials: Maximum number of trials
            replicates: Number of replicates per trial
            timeout: Timeout in seconds
            
        Returns:
            List of trial results
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
        Get the best parameters found.
        
        Returns:
            Dictionary of best parameter values
        """
        best_trial = self.strategy.get_best_trial_from_study(self.study)
        return best_trial.params if best_trial else {}

    def get_best_trial(self) -> Optional[TrialResult]:
        """
        Get the best trial result.
        
        Returns:
            Best trial result, or None if no trials completed
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
