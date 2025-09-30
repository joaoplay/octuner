import time
import threading
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..tunable.mixin import is_llm_tunable, get_call_method_name

logger = logging.getLogger(__name__)


class MethodPatcher:
    """
    Patches LLM methods to measure latency and cost.
    """
    
    def __init__(self):
        self._patched_methods: Dict[int, Tuple[Any, str, Any]] = {}
        self._locks: Dict[int, threading.Lock] = {}
        self._call_logs: Dict[int, List[Dict[str, Any]]] = {}
    
    @contextmanager
    def patch_component(self, component: Any):
        """
        Context manager to patch a component's LLM methods.
        
        Args:
            component: Component to patch
            
        Yields:
            The patched component
        """
        if not is_llm_tunable(component):
            yield component
            return
        
        component_id = id(component)
        method_name = get_call_method_name(component)
        
        try:
            self._patch_method(component, method_name, component_id)
            yield component
        finally:
            self._unpatch_method(component_id)
    
    def _patch_method(self, component: Any, method_name: str, component_id: int):
        """
        Patch a method on a component.
        
        Args:
            component: Component to patch
            method_name: Name of the method to patch
            component_id: Unique ID for the component
        """
        if not hasattr(component, method_name):
            logger.warning(f"Method {method_name} not found on component {component}")
            return
        
        original_method = getattr(component, method_name)
        if not callable(original_method):
            logger.warning(f"Attribute {method_name} is not callable on component {component}")
            return
        
        # Create a lock for this component
        if component_id not in self._locks:
            self._locks[component_id] = threading.Lock()
        
        # Create the wrapped method
        def wrapped_method(*args, **kwargs):
            start_time = time.time()
            result = None
            error = None
            metadata = {}
            
            try:
                # Call the original method
                result = original_method(*args, **kwargs)
                
                # Try to extract token information if available
                if hasattr(result, 'usage') and result.usage:
                    metadata['input_tokens'] = getattr(result.usage, 'prompt_tokens', None)
                    metadata['output_tokens'] = getattr(result.usage, 'completion_tokens', None)
                elif isinstance(result, dict):
                    metadata['input_tokens'] = result.get('usage', {}).get('prompt_tokens')
                    metadata['output_tokens'] = result.get('usage', {}).get('completion_tokens')
                
                return result
                
            except Exception as e:
                error = str(e)
                raise
            finally:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Log the call
                with self._locks[component_id]:
                    if component_id not in self._call_logs:
                        self._call_logs[component_id] = []
                    
                    call_log = {
                        'timestamp': start_time,
                        'latency_ms': latency_ms,
                        'input_tokens': metadata.get('input_tokens'),
                        'output_tokens': metadata.get('output_tokens'),
                        'metadata': metadata,
                        'error': error
                    }
                    
                    self._call_logs[component_id].append(call_log)
        
        # Store the original method and patch info
        self._patched_methods[component_id] = (component, method_name, original_method)
        
        # Apply the patch
        setattr(component, method_name, wrapped_method)
        
        logger.debug(f"Patched method {method_name} on component {component_id}")
    
    def _unpatch_method(self, component_id: int):
        """
        Restore the original method on a component.
        
        Args:
            component_id: ID of the component to unpatch
        """
        if component_id not in self._patched_methods:
            return
        
        component, method_name, original_method = self._patched_methods[component_id]
        
        # Restore the original method
        setattr(component, method_name, original_method)
        
        # Clean up
        del self._patched_methods[component_id]
        if component_id in self._locks:
            del self._locks[component_id]
        
        logger.debug(f"Unpatched method {method_name} on component {component_id}")
    
    def get_call_logs(self, component: Any) -> List[Dict[str, Any]]:
        """
        Get call logs for a component.
        
        Args:
            component: Component to get logs for
            
        Returns:
            List of call log dictionaries
        """
        component_id = id(component)
        return self._call_logs.get(component_id, []).copy()
    
    def clear_call_logs(self, component: Any = None):
        """
        Clear call logs for a component or all components.
        
        Args:
            component: Component to clear logs for, or None for all
        """
        if component is None:
            self._call_logs.clear()
        else:
            component_id = id(component)
            if component_id in self._call_logs:
                del self._call_logs[component_id]
    
    def get_aggregated_metrics(self, component: Any) -> Dict[str, float]:
        """
        Get aggregated metrics for a component.
        
        Args:
            component: Component to get metrics for
            
        Returns:
            Dictionary of aggregated metrics
        """
        logs = self.get_call_logs(component)
        if not logs:
            return {}
        
        # Calculate latency statistics
        latencies = [log['latency_ms'] for log in logs if log['latency_ms'] is not None]
        latency_stats = {}
        if latencies:
            latency_stats = {
                'latency.total_ms': sum(latencies),
                'latency.avg_ms': sum(latencies) / len(latencies),
                'latency.min_ms': min(latencies),
                'latency.max_ms': max(latencies),
                'latency.p50_ms': sorted(latencies)[len(latencies) // 2],
                'latency.p95_ms': sorted(latencies)[int(len(latencies) * 0.95)],
                'latency.p99_ms': sorted(latencies)[int(len(latencies) * 0.99)],
            }
        
        # Calculate cost statistics if available
        cost_stats = {}
        if is_llm_tunable(component) and hasattr(component, 'llm_eq_cost'):
            total_cost = 0
            for log in logs:
                if log['input_tokens'] is not None and log['output_tokens'] is not None:
                    cost = component.llm_eq_cost(
                        input_tokens=log['input_tokens'],
                        output_tokens=log['output_tokens'],
                        metadata=log['metadata']
                    )
                    if cost is not None:
                        total_cost += cost
            
            if total_cost > 0:
                cost_stats = {
                    'cost.total': total_cost,
                    'cost.avg_per_call': total_cost / len(logs),
                }
        
        # Combine all stats
        stats = {
            'calls.total': len(logs),
            'calls.successful': len([log for log in logs if log['error'] is None]),
            'calls.failed': len([log for log in logs if log['error'] is not None]),
        }
        stats.update(latency_stats)
        stats.update(cost_stats)
        
        return stats


# Global patcher instance
_global_patcher = MethodPatcher()


def patch_component(component: Any):
    """
    Convenience function to patch a component.
    
    Args:
        component: Component to patch
        
    Returns:
        Context manager for the patch
    """
    return _global_patcher.patch_component(component)


def get_call_logs(component: Any) -> List[Dict[str, Any]]:
    """
    Get call logs for a component.
    
    Args:
        component: Component to get logs for
        
    Returns:
        List of call log dictionaries
    """
    return _global_patcher.get_call_logs(component)


def get_aggregated_metrics(component: Any) -> Dict[str, float]:
    """
    Get aggregated metrics for a component.
    
    Args:
        component: Component to get metrics for
        
    Returns:
        Dictionary of aggregated metrics
    """
    return _global_patcher.get_aggregated_metrics(component)


def clear_call_logs(component: Any = None):
    """
    Clear call logs for a component or all components.
    
    Args:
        component: Component to clear logs for, or None for all
    """
    _global_patcher.clear_call_logs(component)
