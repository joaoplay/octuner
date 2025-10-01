# Getting Started

Get started with Octuner in 3 simple steps: Configure, Build, Optimize.

## Quick Example

```python
from octuner import MultiProviderTunableLLM, AutoTuner, apply_best

# 1. Create a tunable LLM
llm = MultiProviderTunableLLM(config_file="config_templates/openai_basic.yaml")

# 2. Use it
response = llm.call("What is the capital of France?")
print(response.text)

# 3. Optimize it (when you have a dataset)
dataset = [
    {"input": "What is 2+2?", "target": "4"},
    {"input": "What is 3+3?", "target": "6"},
]

tuner = AutoTuner(
    component=llm,
    entrypoint=lambda llm, inp: llm.call(inp).text,
    dataset=dataset,
    metric=lambda output, target: 1.0 if target in output else 0.0
)

result = tuner.search(n_trials=10)
result.save_best("optimized_config.yaml")
```

### 1. Copy a Config Template

Octuner uses explicit YAML configuration files. Start by copying a template:

```bash
# For OpenAI only
cp config_templates/openai_basic.yaml my_config.yaml

# For multiple providers
cp config_templates/multi_provider.yaml my_config.yaml
```

Set your API keys:

```bash
export OPENAI_API_KEY="sk-your-key"
export GOOGLE_API_KEY="your-key"  # if using Gemini
```

### 2. Build Your Component

Create a simple component that uses the tunable LLM:

```python
from octuner import MultiProviderTunableLLM

class SentimentAnalyzer:
    def __init__(self, config_file: str):
        self.llm = MultiProviderTunableLLM(config_file=config_file)
    
    def analyze(self, text: str) -> str:
        prompt = f"Classify the sentiment (positive/negative/neutral): {text}"
        response = self.llm.call(prompt)
        return response.text.strip().lower()

# Use it
analyzer = SentimentAnalyzer("my_config.yaml")
result = analyzer.analyze("I love this product!")
print(result)  # "positive"
```

### 3. Optimize

When you have a dataset of examples, optimize your component:

```python
from octuner import AutoTuner, apply_best

# Prepare dataset
dataset = [
    {"input": "I love this!", "target": "positive"},
    {"input": "This is terrible", "target": "negative"},
    {"input": "It's okay", "target": "neutral"},
]

# Define metric
def accuracy_metric(output: str, target: str) -> float:
    return 1.0 if target in output else 0.0

# Create optimizer
analyzer = SentimentAnalyzer("my_config.yaml")
tuner = AutoTuner(
    component=analyzer,
    entrypoint=lambda analyzer, text: analyzer.analyze(text),
    dataset=dataset,
    metric=accuracy_metric
)

# Run optimization
result = tuner.search(n_trials=20)
print(f"Best quality: {result.best_trial.metrics.quality:.2f}")

# Save and apply
result.save_best("optimized.yaml")
apply_best(analyzer, "optimized.yaml")
```

That's it! Your component will now use optimized parameters for provider, model, temperature, and other settings.

### What Gets Optimized?

Octuner automatically discovers and tunes:

- **Provider & Model**: OpenAI vs Gemini, GPT-4 vs GPT-3.5, etc.
- **Temperature**: Creativity vs consistency
- **Max Tokens**: Response length
- **Other parameters**: top_p, frequency_penalty, etc.

All defined in your YAML config file.

## Adding Custom Providers

Octuner is designed to be extensible. You can add your own custom providers for self-hosted LLMs (like Ollama, vLLM, or LM Studio) or any other LLM service.

### How It Works

1. **Create a Provider Class** - Inherit from `BaseLLMProvider`
2. **Register the Provider** - Use `register_provider()` function
3. **Configure It** - Create a YAML config with models and parameters
4. **Use It** - Reference it in your `MultiProviderTunableLLM`

### Step-by-Step Example: Ollama Provider

Let's create a custom provider for [Ollama](https://ollama.ai), a self-hosted LLM runtime.

#### 1. Create the Provider Class

```python
# my_custom_provider.py
import time
from typing import Any, Optional
import requests
from octuner.providers import BaseLLMProvider, LLMResponse

class OllamaProvider(BaseLLMProvider):
    """Custom provider for Ollama self-hosted LLMs."""
    
    def __init__(self, config_loader, **kwargs):
        super().__init__(config_loader=config_loader, **kwargs)
        self.provider_name = "ollama"
        self.base_url = kwargs.get('base_url', 'http://localhost:11434')
    
    def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """Make a call to Ollama API."""
        start_time = time.time()
        
        # Get model from kwargs or config
        model = self._get_parameter("model", kwargs, "llama2")
        
        # Build request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Add supported parameters
        supported_params = self.config_loader.get_supported_parameters(
            self.provider_name, model
        )
        
        for param in supported_params:
            param_value = self._get_parameter(param, kwargs, model)
            if param_value is not None:
                converted_value = self._convert_parameter_type(param, param_value, model)
                request_data[param] = converted_value
        
        # Make request
        response = self._make_request(**request_data)
        return self._parse_response(response, start_time, model)
    
    def _make_request(self, **kwargs) -> Any:
        """Make the actual API request to Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=kwargs,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    def _parse_response(self, response: Any, start_time: float, model: str) -> LLMResponse:
        """Parse Ollama response into LLMResponse."""
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            text=response.get("response", ""),
            provider="ollama",
            model=model,
            cost=0.0,  # Self-hosted = free
            latency_ms=latency_ms,
            metadata=response
        )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost (free for self-hosted)."""
        return 0.0
```

#### 2. Register the Provider

Use the `register_provider()` function for clean registration:

```python
from octuner import register_provider
from my_custom_provider import OllamaProvider

# Register the custom provider - clean and simple!
register_provider('ollama', OllamaProvider)
```

You can also list available providers:

```python
from octuner import list_providers

print(list_providers())  # ['openai', 'gemini', 'ollama']
```

#### 3. Create a Config File

Create `ollama_config.yaml`:

```yaml
providers:
  ollama:
    default_model: "llama2"
    
    available_models:
      - llama2
      - mistral
      - codellama
    
    # Free for self-hosted
    pricing_usd_per_1m_tokens:
      llama2: [0.0, 0.0]
      mistral: [0.0, 0.0]
      codellama: [0.0, 0.0]
    
    model_capabilities:
      llama2:
        supported_parameters: [temperature, top_p, top_k]
        parameters:
          temperature:
            type: float
            range: [0.0, 1.0]
            default: 0.7
          top_p:
            type: float
            range: [0.0, 1.0]
            default: 0.9
          top_k:
            type: int
            range: [1, 100]
            default: 40
        forced_parameters: {}
      
      mistral:
        supported_parameters: [temperature, top_p, top_k]
        parameters:
          temperature:
            type: float
            range: [0.0, 1.0]
            default: 0.7
          top_p:
            type: float
            range: [0.0, 1.0]
            default: 0.9
          top_k:
            type: int
            range: [1, 100]
            default: 40
        forced_parameters: {}
      
      codellama:
        supported_parameters: [temperature, top_p, top_k]
        parameters:
          temperature:
            type: float
            range: [0.0, 1.0]
            default: 0.7
          top_p:
            type: float
            range: [0.0, 1.0]
            default: 0.9
          top_k:
            type: int
            range: [1, 100]
            default: 40
        forced_parameters: {}
```

#### 4. Use Your Custom Provider

```python
from octuner import MultiProviderTunableLLM, AutoTuner, register_provider
from my_custom_provider import OllamaProvider

# Register custom provider with clean API
register_provider('ollama', OllamaProvider)

# Create tunable LLM with your custom provider
llm = MultiProviderTunableLLM(
    config_file="ollama_config.yaml",
    default_provider="ollama",
    provider_configs={
        "ollama": {"base_url": "http://localhost:11434"}
    }
)

# Use it
response = llm.call("What is machine learning?")
print(response.text)

# Optimize it
dataset = [
    {"input": "Explain AI", "target": "artificial intelligence"},
    {"input": "What is ML?", "target": "machine learning"},
]

tuner = AutoTuner(
    component=llm,
    entrypoint=lambda llm, inp: llm.call(inp).text,
    dataset=dataset,
    metric=lambda output, target: 1.0 if target.lower() in output.lower() else 0.0
)

result = tuner.search(n_trials=10)
print(f"Best config: {result.best_trial.params}")
```

### Key Implementation Requirements

When creating a custom provider, you **must** implement:

1. **`__init__(self, config_loader, **kwargs)`** - Initialize with config loader
2. **`provider_name`** - Set a unique string identifier
3. **`call(prompt, system_prompt, **kwargs)`** - Main entry point for LLM calls
4. **`_make_request(**kwargs)`** - Handle the actual API request
5. **`_parse_response(response)`** - Parse API response to `LLMResponse`
6. **`_calculate_cost(input_tokens, output_tokens, model)`** - Calculate costs

The base class provides:
- `_get_parameter()` - Get parameters from config or kwargs
- `_convert_parameter_type()` - Automatic type conversion
- `get_cost_per_token()` - Get pricing from config
- `get_available_models()` - Get models from config

### Provider Registration API

Octuner provides clean functions for managing providers:

```python
from octuner import (
    register_provider,      # Add a custom provider
    unregister_provider,    # Remove a provider
    list_providers          # List all registered providers
)

# Register
register_provider('ollama', OllamaProvider)

# Check what's available
print(list_providers())  # ['openai', 'gemini', 'ollama']

# Unregister if needed
unregister_provider('ollama')
```

### Multi-Provider Optimization

You can even optimize across cloud and self-hosted providers:

```yaml
# multi_config.yaml
providers:
  openai:
    # ... OpenAI config ...
  
  ollama:
    # ... Ollama config ...
```

```python
from octuner import MultiProviderTunableLLM, register_provider

# Register custom provider
register_provider('ollama', OllamaProvider)

# Octuner will now optimize across both!
llm = MultiProviderTunableLLM(config_file="multi_config.yaml")
result = tuner.search(n_trials=50)  # Test both OpenAI and Ollama
```

This allows you to discover whether a self-hosted model can match cloud performance for your specific use case!

## Next Steps

- **[API Reference](reference.md)** - Complete API documentation and detailed examples
- **[Contributing](contributing.md)** - Learn how to contribute to Octuner
- **[Installation](installation.md)** - Review installation and setup options

---

**Key Principle**: Octuner is library-first. You explicitly provide config files, no global defaults or hidden state.