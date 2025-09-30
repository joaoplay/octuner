"""
Execution engine for Octuner.

Runs the entrypoint function over the dataset and collects metrics.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from ..tunable.types import Dataset, DatasetItem, MetricFunction, EntrypointFunction, MetricResult
from ..utils.patcher import patch_component, get_aggregated_metrics, clear_call_logs

logger = logging.getLogger(__name__)


class DatasetExecutor:
    """
    Executes the entrypoint function over a dataset and collects metrics.
    """
    
    def __init__(self, component: Any, entrypoint: EntrypointFunction, dataset: Dataset, metric: MetricFunction,
                 max_workers: int = 1):
        """
        Initialize the executor.
        
        Args:
            component: Component to tune
            entrypoint: Function to call with (component, input)
            dataset: Dataset of input/target pairs
            metric: Function to compute quality score
            max_workers: Maximum number of concurrent workers
        """
        self.component = component
        self.entrypoint = entrypoint
        self.dataset = dataset
        self.metric = metric
        self.max_workers = max_workers
    
    def execute_trial(self, parameters: Dict[str, Any]) -> MetricResult:
        """
        Execute a single trial with given parameters.
        
        Args:
            parameters: Parameter values to set
            
        Returns:
            Aggregated metrics for the trial
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
                if quality_scores:
                    quality_scores.sort()
                    median_quality = quality_scores[len(quality_scores) // 2]
                else:
                    median_quality = 0.0
                
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
        Execute the dataset in parallel.
        
        Returns:
            List of quality scores
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
        Execute a trial multiple times and aggregate results.
        
        Args:
            parameters: Parameter values to set
            replicates: Number of replicates to run
            
        Returns:
            Aggregated metrics across replicates
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
        if qualities:
            qualities.sort()
            median_quality = qualities[len(qualities) // 2]
        else:
            median_quality = 0.0
        
        total_cost = sum(costs) if costs else None
        
        if latencies:
            latencies.sort()
            median_latency = latencies[len(latencies) // 2]
        else:
            median_latency = None
        
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
        Execute the full dataset with given parameters.
        
        Args:
            parameters: Parameter values to set
            
        Returns:
            List of MetricResult for each dataset item
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

    def _calculate_aggregate_metrics(self, results: List[MetricResult]) -> MetricResult:
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
        if qualities:
            qualities.sort()
            median_quality = qualities[len(qualities) // 2]
        else:
            median_quality = 0.0
        
        total_cost = sum(costs) if costs else 0.0
        
        if latencies:
            latencies.sort()
            median_latency = latencies[len(latencies) // 2]
        else:
            median_latency = 0.0
        
        return MetricResult(quality=median_quality, cost=total_cost, latency_ms=median_latency)


def execute_trial(component: Any, entrypoint: EntrypointFunction, dataset: Dataset, metric: MetricFunction,
                  parameters: Dict[str, Any], max_workers: int = 1, replicates: int = 1, trial_number: int = None) -> MetricResult:
    """
    Convenience function to execute a single trial.
    
    Args:
        component: Component to tune
        entrypoint: Function to call with (component, input)
        dataset: Dataset of input/target pairs
        metric: Function to compute quality score
        parameters: Parameter values to set
        max_workers: Maximum number of concurrent workers
        replicates: Number of replicates to run
        
    Returns:
        Aggregated metrics for the trial
    """
    executor = DatasetExecutor(component, entrypoint, dataset, metric, max_workers)
    return executor.execute_with_replicates(parameters, replicates)
