import hashlib
import logging
import time
from typing import Any, Dict, Optional
import yaml
from .setter import set_parameters

logger = logging.getLogger(__name__)


def save_parameters_to_yaml(parameters: Dict[str, Any],  path: str,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save parameters to a YAML file.
    
    Args:
        parameters: Dictionary of parameter values
        path: Path to save the YAML file
        metadata: Optional metadata to include
    """
    # Prepare the data structure
    data = {
        "parameters": parameters,
        "metadata": {
            "timestamp": time.time(),
            **(metadata or {})
        }
    }
    
    # Add metrics_summary if provided in metadata
    if metadata and "metrics_summary" in metadata:
        data["metrics_summary"] = metadata["metrics_summary"]
    elif metadata and any(key in metadata for key in ["quality", "cost", "latency_ms"]):
        # If metrics are provided directly in metadata, create metrics_summary
        metrics_summary = {}
        for key in ["quality", "cost", "latency_ms"]:
            if key in metadata:
                metrics_summary[key] = metadata[key]
        data["metrics_summary"] = metrics_summary
    else:
        # If no metrics provided, set to None
        data["metrics_summary"] = None
    
    # Add dataset fingerprint if available
    if metadata and "dataset_fingerprint" in metadata:
        data["metadata"]["dataset_fingerprint"] = metadata["dataset_fingerprint"]
    
    # Write to file
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved {len(parameters)} parameters to {path}")


def load_parameters_from_yaml(path: str) -> Dict[str, Any]:
    """
    Load parameters from a YAML file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Dictionary of parameter values
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file is invalid YAML
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise yaml.YAMLError("Invalid YAML structure: expected dictionary")
    
    # Handle both old and new format
    if "parameters" in data:
        parameters = data["parameters"]
    elif "params" in data:
        parameters = data["params"]
    else:
        raise yaml.YAMLError("Invalid YAML structure: missing 'parameters' or 'params' section")
    
    if not isinstance(parameters, dict):
        raise yaml.YAMLError("Invalid YAML structure: 'parameters' must be a dictionary")
    
    logger.info(f"Loaded {len(parameters)} parameters from {path}")
    return parameters


def apply_best(component: Any, path_or_params) -> None:
    """
    Apply the best parameters to a component.
    
    This function safely sets the optimal parameters found during tuning
    on a fresh component instance for production use.
    
    Args:
        component: Component to apply parameters to
        path_or_params: Either a path to YAML file (str) or parameters dict/object
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
        ValueError: If any parameter values are invalid
        AttributeError: If any parameter paths don't exist
    """
    try:
        # Handle different input types
        if isinstance(path_or_params, str):
            # Load parameters from YAML file
            parameters = load_parameters_from_yaml(path_or_params)
        elif isinstance(path_or_params, dict):
            # Use parameters dict directly
            parameters = path_or_params
        elif hasattr(path_or_params, 'best_parameters'):
            # Handle SearchResult or similar objects
            parameters = path_or_params.best_parameters
        elif hasattr(path_or_params, 'parameters'):
            # Handle TrialResult or similar objects
            parameters = path_or_params.parameters
        else:
            raise ValueError(f"Unsupported parameter type: {type(path_or_params)}")
        
        # Apply parameters
        set_parameters(component, parameters)
        
        logger.info(f"Successfully applied {len(parameters)} parameters to component")
        
        # Log the applied parameters for verification
        for param_path, value in parameters.items():
            logger.debug(f"Applied {param_path} = {value}")
            
    except Exception as e:
        logger.error(f"Failed to apply best parameters: {e}")
        raise


def compute_dataset_fingerprint(dataset: Any) -> str:
    """
    Compute a fingerprint for a dataset to track which dataset
    was used for tuning.
    
    Args:
        dataset: Dataset to fingerprint
        
    Returns:
        SHA256 hash string
    """
    # Convert dataset to a string representation
    if hasattr(dataset, '__len__'):
        # For list-like datasets, use length and first few items
        dataset_str = f"len:{len(dataset)}"
        if len(dataset) > 0:
            dataset_str += f",first:{str(dataset[0])[:100]}"
        if len(dataset) > 1:
            dataset_str += f",last:{str(dataset[-1])[:100]}"
    else:
        # For other types, use string representation
        dataset_str = str(dataset)
    
    # Compute hash
    hash_obj = hashlib.sha256(dataset_str.encode('utf-8'))
    return f"sha256:{hash_obj.hexdigest()[:16]}"


def create_metadata_summary(trials: list, optimization_mode: str = "pareto",
    dataset_size: int = 0, total_trials: int = 0, best_quality: float = 0.0,
    best_cost: Optional[float] = None, best_latency_ms: Optional[float] = None,
    dataset_fingerprint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a metadata summary for the optimization results.
    
    Args:
        trials: List of trial results
        optimization_mode: Mode used for optimization
        dataset_size: Number of examples in the dataset
        total_trials: Total number of trials run
        best_quality: Best quality score achieved
        best_cost: Best cost achieved (if available)
        best_latency_ms: Best latency achieved (if available)
        dataset_fingerprint: Dataset fingerprint (if available)
        
    Returns:
        Dictionary of metadata
    """
    # If trials are provided, extract metrics from them
    if trials:
        successful_trials = [t for t in trials if hasattr(t, 'success') and t.success]
        failed_trials = [t for t in trials if hasattr(t, 'success') and not t.success]
        
        # Calculate metrics from all trials (not just successful ones)
        all_qualities = [t.metrics.quality for t in trials if hasattr(t.metrics, 'quality')]
        all_costs = [t.metrics.cost for t in trials if hasattr(t.metrics, 'cost') and t.metrics.cost is not None]
        all_latencies = [t.metrics.latency_ms for t in trials if hasattr(t.metrics, 'latency_ms') and t.metrics.latency_ms is not None]
        
        # For avg_quality, use successful trials only if there are failures
        if failed_trials:
            # Use only successful trials for avg_quality when there are failures
            successful_qualities = [t.metrics.quality for t in successful_trials if hasattr(t.metrics, 'quality')]
            if successful_qualities:
                avg_quality = round(sum(successful_qualities) / len(successful_qualities), 3)
            else:
                avg_quality = 0.0
        else:
            # Use all trials for avg_quality when all are successful
            if all_qualities:
                avg_quality = round(sum(all_qualities) / len(all_qualities), 3)
            else:
                avg_quality = 0.0
        
        if all_qualities:
            best_quality = max(all_qualities)
        else:
            best_quality = 0.0
            
        if all_costs:
            best_cost = min(all_costs)
            total_cost = sum(all_costs)
        else:
            best_cost = None
            total_cost = 0.0
            
        if all_latencies:
            best_latency_ms = min(all_latencies)
            avg_latency_ms = sum(all_latencies) / len(all_latencies)
        else:
            best_latency_ms = None
            avg_latency_ms = 0.0
        
        total_trials = len(trials)
        successful_count = len(successful_trials)
    else:
        total_trials = 0
        successful_count = 0
        avg_quality = 0.0
        best_quality = 0.0
        total_cost = 0.0
        avg_latency_ms = 0.0
    
    # Calculate success rate
    success_rate = successful_count / total_trials if total_trials > 0 else 0.0
    
    metadata = {
        "mode": optimization_mode,
        "dataset_size": dataset_size,
        "total_trials": total_trials,
        "successful_trials": successful_count,
        "success_rate": success_rate,  # Don't round to preserve exact fractions
        "avg_quality": round(avg_quality, 3),
        "best_quality": round(best_quality, 3),
        "total_cost": round(total_cost, 4),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "metrics_summary": {
            "quality.overall": round(best_quality, 3)
        }
    }
    
    if best_cost is not None:
        metadata["metrics_summary"]["cost.total_per_ex"] = round(best_cost, 4)
    
    if best_latency_ms is not None:
        metadata["metrics_summary"]["latency.ttl_ms_p95"] = round(best_latency_ms, 0)
    
    if dataset_fingerprint:
        metadata["dataset_fingerprint"] = dataset_fingerprint
    
    return metadata


def validate_parameters_file(path: str) -> bool:
    """
    Validate that a parameters file has the correct structure.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            return False
        
        if "params" not in data:
            return False
        
        params = data["params"]
        if not isinstance(params, dict):
            return False
        
        # Check that all parameter values are basic types
        for key, value in params.items():
            if not isinstance(key, str):
                return False
            
            if not isinstance(value, (str, int, float, bool)):
                return False
        
        return True
        
    except Exception:
        return False
