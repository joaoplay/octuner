# Octuner

A lightweight library that optimizes LLM parameters via explicit YAML configuration and tuning against golden datasets.

## Features

- **YAML-First Configuration**: Explicit parameter control via configuration files
- **Multi-Provider Support**: Optimize across OpenAI, Gemini, and other providers
- **Parameter tuning**: Optimizes only explicitly declared tunable parameters  
- **Multiple optimization modes**: Pareto, constrained, and scalarized optimization
- **Cost & latency tracking**: Optional measurement of performance metrics
- **Production ready**: Copy and customize configuration templates

## Quick Start

### 1. Copy a Configuration Template

```bash
# Copy a starter template
cp config_templates/openai_basic.yaml my_llm_config.yaml

# Set your API key
export OPENAI_API_KEY=sk-your-key-here
```

### 2. Use in Your Code

```python
from octuner import MultiProviderTunableLLM

# Explicit configuration - no hidden global state
llm = MultiProviderTunableLLM(config_file="my_llm_config.yaml")
response = llm.call("What is the capital of France?")
print(response.text)
```

### 3. Define What Gets Optimized (in YAML)

```yaml
providers:
  openai:
    model_capabilities:
      gpt-4o-mini:
        # Explicitly declare what parameters to optimize
        supported_parameters: [temperature, top_p, max_tokens]
        parameter_ranges:
          temperature: [0.0, 2.0]
          max_tokens: [50, 4000]
        default_parameters:
          temperature: 0.7
          max_tokens: 1000
```

## Configuration Templates

Choose from ready-to-use templates in `config_templates/`:

- **`openai_basic.yaml`** - Basic OpenAI setup (GPT-3.5, GPT-4o, GPT-4o-mini)
- **`gemini_basic.yaml`** - Basic Gemini setup (cost-effective)
- **`multi_provider.yaml`** - Multiple providers (let optimization choose)
- **`task_specific.yaml`** - Task-specific model variants

## Advanced Usage

### Task-Specific Model Variants

For most use cases, you'll want to use the built-in `MultiProviderTunableLLM` which automatically discovers parameters from YAML configuration files:

```python
from octuner import MultiProviderTunableLLM

# Create a tunable LLM that discovers parameters from config
llm = MultiProviderTunableLLM(
    config_file="config_templates/openai_basic.yaml",
    default_provider="openai",
    provider_configs={"openai": {"api_key": "your-key"}}
)

# The system automatically registers tunable parameters based on the YAML config
# No manual registration needed!
```

For custom components, you can register them as tunable programmatically:

```python
from octuner import TunableMixin, register_tunable_class

class MyCustomLLM(TunableMixin):
    def __init__(self):
        # Register this class as tunable
        register_tunable_class(
            self.__class__,
            params={
                "temperature": ("float", 0.0, 1.0),
                "top_p": ("float", 0.0, 1.0),
                "max_tokens": ("int", 64, 4096),
                "model_id": ("choice", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]),
            },
            call_method="send_prompt"
        )
    
    def llm_eq_cost(self, *, input_tokens=None, output_tokens=None, metadata=None):
        if input_tokens is None or output_tokens is None:
            return None
        # Calculate cost in your currency
        in_price, out_price = 1.5, 5.0  # EUR / 1M tokens
        return (input_tokens/1e6)*in_price + (output_tokens/1e6)*out_price
```

### 2. Tune Your Component

```python
from octuner import AutoTuner, apply_best
from myapp.slips import SlipEvaluator

def metric(output: dict, target: dict) -> float:
    """Quality metric function (0.0 to 1.0)."""
    # Your custom metric logic here
    return 1.0 if output == target else 0.0

# Prepare your dataset
dataset = [
    {"input": input_obj1, "target": expected_output1},
    {"input": input_obj2, "target": expected_output2},
    # ... more examples
]

# Create tuner
tuner = AutoTuner.from_component(
    component=SlipEvaluator(),
    entrypoint=lambda se, slip: se.analyse(slip),
    dataset=dataset,
    metric=metric,
)

# Optional: Filter parameters
tuner.include(["*.temperature", "*.top_p", "*.max_tokens", "*.model_id"]) \
     .exclude(["*.verbose", "*.debug"])

# Run optimization
result = tuner.search(
    max_trials=120,
    mode="pareto",  # or "constrained" or "scalarized"
    replicates=3    # Run each trial 3 times for stability
)

# Save best parameters
result.save_best("best_slip.yaml")

# Apply in production
se = SlipEvaluator()
apply_best(se, "best_slip.yaml")
final_result = se.analyse(betting_slip)
```

## Core Concepts

### TunableMixin Mixin

The `TunableMixin` mixin class defines what makes a component tunable:

**Registry Configuration:**
- `register_tunable_class(cls, params, call_method, proxy)`: Register a class as tunable
  - `params`: Dictionary mapping parameter names to (type, low, high) or (type, choices)  
  - `call_method`: Method name to wrap for timing/cost measurement (default: "send_prompt")
  - `proxy`: Optional proxy attribute if tunables live on a nested object
- `llm_eq_cost()`: Optional method to calculate per-call cost

### Parameter Types

- **float**: Continuous values with min/max bounds
- **int**: Integer values with min/max bounds  
- **choice**: Categorical values from a predefined list
- **bool**: Boolean values

### Optimization Modes

- **Pareto** (default): Multi-objective optimization of quality, cost, and latency
- **Constrained**: Maximize quality with hard constraints on cost/latency
- **Scalarized**: Minimize weighted combination of objectives

## Advanced Usage

### Constrained Optimization

```python
result = tuner.search(
    mode="constrained",
    constraints={
        "latency_ms": 1000,      # Max 1 second
        "cost_total": 0.01       # Max 1 cent
    }
)
```

### Scalarized Optimization

```python
from octuner.tunable.types import ScalarizationWeights

weights = ScalarizationWeights(
    cost_weight=100.0,      # Cost is 100x more important than quality
    latency_weight=0.1      # Latency is 0.1x as important as quality
)

result = tuner.search(
    mode="scalarized",
    scalarization_weights=weights
)
```

### Custom Discovery Filters

```python
# Only tune temperature and top_p parameters
tuner.include(["*.temperature", "*.top_p"])

# Exclude verbose/debug parameters
tuner.exclude(["*.verbose", "*.debug", "*.log_level"])
```

### Parallel Execution

```python
# Use multiple workers for dataset evaluation
tuner = AutoTuner.from_component(
    component=component,
    entrypoint=entrypoint,
    dataset=dataset,
    metric=metric,
    max_workers=4  # Parallel execution
)
```

## Output Format

The `save_best()` method creates a YAML file like this:

```yaml
params:
  data_gathering_llm.primary_llm.temperature: 0.18
  refinement_llm.primary_llm.temperature: 0.12
  reasoning_llm.primary_llm.max_tokens: 448
  data_gathering_llm.primary_llm.model_id: gpt-4o-mini

meta:
  dataset_fingerprint: "sha256:abc123..."
  mode: "pareto"
  metrics_summary:
    quality.overall: 0.84
    latency.ttl_ms_p95: 1180
    cost.total_per_ex: 0.0069
```

## Installation

```bash
pip install octuner
```


## Development

```bash
git clone https://github.com/yourusername/octuner.git
cd octuner
pip install -e ".[dev]"
pytest
```

## Requirements

- Python 3.10+
- Optuna 3.0+
- PyYAML 6.0+

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please check the issues and pull requests for current discussions.
