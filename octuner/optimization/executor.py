import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from ..tunable.types import Dataset, DatasetItem, MetricFunction, EntrypointFunction, MetricResult
from ..utils.patcher import patch_component, get_aggregated_metrics, clear_call_logs

logger = logging.getLogger(__name__)


class DatasetExecutor:
    """
    DatasetExecutor is a core component of the optimization system that
    handles the execution of evaluation trials during parameter optimization.
    It manages the evaluation of components over datasets, collects performance
    metrics, and provides both sequential and parallel execution capabilities.

    How it works:

    1. **Parameter Application**: Applies trial parameters to the component using
       the parameter setter utilities, ensuring consistent parameter configuration
       across trials.

    2. **Dataset Evaluation**: Executes the entrypoint function over each dataset
       item, collecting outputs and computing quality scores using the provided
       metric function.

    3. **Metrics Collection**: Automatically collects comprehensive metrics including
       quality scores, execution costs, and latency measurements for each trial.

    4. **Parallel Execution**: Supports concurrent evaluation of dataset items
       using ThreadPoolExecutor for faster trial execution when max_workers > 1.

    5. **Statistical Aggregation**: Provides robust statistical aggregation across
       replicates and dataset items using median-based aggregation for stability.

    Key Features:

    - **Parallel Execution**: Multi-threaded evaluation for faster optimization
    - **Comprehensive Metrics**: Quality, cost, and latency tracking
    - **Statistical Robustness**: Median-based aggregation to handle outliers
    - **Error Handling**: Graceful handling of individual item failures
    - **Replicate Support**: Multiple trial runs for statistical significance
    - **Cost Tracking**: Automatic cost collection from tunable components
    """
    
    def __init__(self, component: Any, entrypoint: EntrypointFunction, dataset: Dataset, metric: MetricFunction,
                 max_workers: int = 1):
        """
        Constructor

        Args:
            component: The component to evaluate. Must be tunable and support parameter
                        setting via the parameter setter utilities.
            entrypoint: Function that evaluates the component with input data.
                       Called as `entrypoint(component, input_data)` for each
                       dataset item. Should return a dictionary or object that
                       can be processed by the metric function.
            dataset: List of input/target pairs for evaluation. Each item should
                    be a dictionary with 'input' and 'target' keys containing
                    the input data and expected output respectively.
            metric: Function that computes quality scores from evaluation results.
                   Called as `metric(output, target)` where output is the result
                   from entrypoint and target is the expected output. Should
                   return a float score (higher is better).
            max_workers: Maximum number of concurrent workers for parallel
                        evaluation. Use 1 for sequential execution, >1 for
                        parallel execution. Higher values speed up I/O-bound
                        tasks but may not help with CPU-bound tasks due to
                        Python's GIL.
        """
        self.component = component
        self.entrypoint = entrypoint
        self.dataset = dataset
        self.metric = metric
        self.max_workers = max_workers
    
    @staticmethod
    def _calculate_median(values: List[float], default: float = 0.0) -> float:
        """
        Calculate the median value from a list of numbers.
        
        Args:
            values: List of numeric values
            default: Default value to return if the list is empty
            
        Returns:
            Median value or default if list is empty
        """
        if not values:
            return default
        sorted_values = sorted(values)
        return sorted_values[len(sorted_values) // 2]
    
    def execute_trial(self, parameters: Dict[str, Any]) -> MetricResult:
        """
        Execute a single evaluation trial with the given parameters. t applies the trial
        parameters to the component, executes the evaluation over all dataset items, and
        returns aggregated metrics including quality, cost, and latency.

        The execution process:

        1. **Parameter Application**: Sets the trial parameters on the component
        2. **Call Log Clearing**: Clears any previous call logs for clean metrics
        3. **Dataset Evaluation**: Runs the entrypoint function over each dataset item
        4. **Metrics Collection**: Collects quality scores, costs, and timing data
        5. **Statistical Aggregation**: Computes median quality and total cost

        Args:
            parameters: Dictionary of parameter values to set on the component.
                      Keys should match the parameter paths from the search space.
                      Example: {"llm.temperature": 0.7, "llm.max_tokens": 100}

        Returns:
            MetricResult containing:
            - quality: Median quality score across all dataset items
            - cost: Total cost from all component calls (if available)
            - latency_ms: Total execution time in milliseconds
        """
        # Set parameters on the component
        from ..utils.setter import set_parameters
        set_parameters(self.component, parameters)
        
        # Clear previous call logs
        clear_call_logs(self.component)
        
        # Execute over the dataset
        start_time = time.time()
        quality_scores = []
        
        try:
            with patch_component(self.component):
                if self.max_workers == 1:
                    # Sequential execution
                    for item in self.dataset:
                        try:
                            output = self.entrypoint(self.component, item["input"])
                            score = self.metric(output, item["target"])
                            quality_scores.append(score)
                        except Exception as e:
                            logger.warning(f"Failed to process item: {e}")
                            quality_scores.append(0.0)
                else:
                    # Parallel execution
                    quality_scores = self._execute_parallel()
                
                end_time = time.time()
                total_latency_ms = (end_time - start_time) * 1000
                
                # Get aggregated metrics from the patcher
                aggregated = get_aggregated_metrics(self.component)
                
                # Calculate quality (median of scores)
                median_quality = self._calculate_median(quality_scores)
                
                # Build result
                result = MetricResult(
                    quality=median_quality,
                    cost=aggregated.get('cost.total'),
                    latency_ms=total_latency_ms
                )
                
                logger.debug(f"Trial completed: quality={median_quality:.3f}, "
                           f"latency={total_latency_ms:.1f}ms, "
                           f"cost={result.cost}")
                
                return result
                
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return MetricResult(quality=0.0, cost=None, latency_ms=None)
    
    def _execute_parallel(self) -> List[float]:
        """
        This method implements parallel execution of dataset items using Python
        ThreadPoolExecutor. It's particularly effective for I/O-bound tasks like
        API calls, file operations, or network requests where the GIL doesn't
        significantly impact performance.

        Returns:
            List of quality scores from all dataset items, in completion order.
            Failed items contribute a score of 0.0.
        """
        quality_scores = []
        
        def process_item(item: DatasetItem) -> float:
            try:
                output = self.entrypoint(self.component, item["input"])
                return self.metric(output, item["target"])
            except Exception as e:
                logger.warning(f"Failed to process item: {e}")
                return 0.0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all items
            future_to_item = {executor.submit(process_item, item): item for item in self.dataset}
            
            # Collect results
            for future in as_completed(future_to_item):
                try:
                    score = future.result()
                    quality_scores.append(score)
                except Exception as e:
                    logger.warning(f"Failed to get result: {e}")
                    quality_scores.append(0.0)
        
        return quality_scores
    
    def execute_with_replicates(self, parameters: Dict[str, Any], replicates: int = 1) -> MetricResult:
        """
        Execute a trial multiple times and aggregate results for statistical robustness.
        It's particularly useful for optimization scenarios where individual trials may
        have high variance due to non-deterministic components or external factors.

        Args:
            parameters: Dictionary of parameter values to set on the component.
                      Same format as execute_trial().
            replicates: Number of times to run the trial. Higher values provide
                       better statistical significance but take longer. Typical
                       values range from 1-10 depending on variance requirements.

        Returns:
            MetricResult containing aggregated metrics across all replicates:
        """
        if replicates == 1:
            return self.execute_trial(parameters)
        
        # Run multiple replicates
        results = []
        for i in range(replicates):
            logger.debug(f"Running replicate {i+1}/{replicates}")
            result = self.execute_trial(parameters)
            results.append(result)
        
        # Aggregate results (median for quality, sum for cost, median for latency)
        qualities = [r.quality for r in results if r.quality is not None]
        costs = [r.cost for r in results if r.cost is not None]
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        
        # Calculate aggregated metrics
        median_quality = self._calculate_median(qualities)
        total_cost = sum(costs) if costs else None
        median_latency = self._calculate_median(latencies) if latencies else None
        
        return MetricResult(quality=median_quality, cost=total_cost, latency_ms=median_latency)

    def _execute_single_item(self, item: DatasetItem) -> MetricResult:
        """
        Execute a single dataset item.
        
        Args:
            item: Single dataset item with 'input' and 'target' keys
            
        Returns:
            MetricResult for the single item
        """
        try:
            start_time = time.time()
            
            # Call the entrypoint function
            output = self.entrypoint(self.component, item["input"])
            
            # Calculate quality score
            quality = self.metric(output, item["target"])
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Get cost from component if available
            cost = getattr(self.component, '_last_cost', None)
            if cost is None:
                cost = 0.0
            
            return MetricResult(
                quality=quality,
                cost=cost,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.warning(f"Failed to execute single item: {e}")
            return MetricResult(quality=0.0, cost=0.0, latency_ms=0.0)

    def execute(self, parameters: Dict[str, Any]) -> List[MetricResult]:
        """
        Execute the full dataset and return per-item results, unlike execute_trial()
        which returns aggregated metrics. It's useful for detailed analysis, debugging,
        r when you need to examine individual item performance rather than overall trial
        performance.

        Args:
            parameters: Dictionary of parameter values to set on the component.
                      Same format as execute_trial().

        Returns:
            List of MetricResult objects, one for each dataset item. Each result
            contains the quality, cost, and latency for that specific item.
        """
        # Set parameters on the component
        from ..utils.setter import set_parameters
        set_parameters(self.component, parameters)
        
        # Clear previous call logs
        from ..utils.patcher import clear_call_logs
        clear_call_logs(self.component)
        
        results = []
        for item in self.dataset:
            result = self._execute_single_item(item)
            results.append(result)
        
        return results

    @staticmethod
    def _calculate_aggregate_metrics(results: List[MetricResult]) -> MetricResult:
        """
        Calculate aggregate metrics from a list of results.
        
        Args:
            results: List of MetricResult objects
            
        Returns:
            Aggregated MetricResult
        """
        if not results:
            return MetricResult(quality=0.0, cost=0.0, latency_ms=0.0)
        
        # Extract metrics
        qualities = [r.quality for r in results if r.quality is not None]
        costs = [r.cost for r in results if r.cost is not None]
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        
        # Calculate aggregated metrics
        median_quality = DatasetExecutor._calculate_median(qualities)
        total_cost = sum(costs) if costs else 0.0
        median_latency = DatasetExecutor._calculate_median(latencies)
        
        return MetricResult(quality=median_quality, cost=total_cost, latency_ms=median_latency)


def execute_trial(component: Any, entrypoint: EntrypointFunction, dataset: Dataset, metric: MetricFunction,
                  parameters: Dict[str, Any], max_workers: int = 1, replicates: int = 1, trial_number: int = None) -> MetricResult:
    """
    Convenience function to execute a single trial without creating a DatasetExecutor.
    """
    executor = DatasetExecutor(component, entrypoint, dataset, metric, max_workers)
    return executor.execute_with_replicates(parameters, replicates)
