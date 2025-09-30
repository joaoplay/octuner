# Octuner Configuration Templates

This directory contains starter YAML configuration files for common LLM providers. Copy and customize these files for your project.

## Quick Start

1. **Copy a template** to your project directory:
   ```bash
   cp config_templates/openai_basic.yaml my_llm_config.yaml
   ```

2. **Use it in your code**:
   ```python
   from octuner import MultiProviderTunableLLM
   
   llm = MultiProviderTunableLLM(config_file="my_llm_config.yaml")
   ```

3. **Set API keys** via environment variables:
   ```bash
   export OPENAI_API_KEY=your_key_here
   export GOOGLE_API_KEY=your_key_here
   
   # For Gemini web search (optional)
   export GOOGLE_SEARCH_API_KEY=your_search_key_here
   export GOOGLE_SEARCH_ENGINE_ID=your_engine_id_here
   ```

## Web Search Setup

### OpenAI Web Search
OpenAI's web search is built-in and requires no additional setup. Just set your `OPENAI_API_KEY`.

### Gemini Web Search
Gemini web search uses Google's native grounding tool. No additional setup required beyond your `GOOGLE_API_KEY`.

**Note**: The grounding tool provides native web search capabilities without requiring additional API keys or configuration.

## Available Templates

### `openai_basic.yaml`
- **Use case**: OpenAI models only (GPT-3.5, GPT-4o, GPT-4o-mini)
- **Parameters**: All OpenAI parameters (temperature, top_p, frequency_penalty, etc.)
- **Best for**: Simple OpenAI-only projects

### `gemini_basic.yaml`
- **Use case**: Google Gemini models only
- **Parameters**: Gemini-supported parameters (temperature, top_p, max_tokens, web search)
- **Web search**: Via Google's native grounding tool
- **Best for**: Cost-conscious projects using Gemini

### `multi_provider.yaml`
- **Use case**: Both OpenAI and Gemini providers
- **Parameters**: Provider-specific parameter sets with web search support
- **Web search**: OpenAI (built-in) + Gemini (Google grounding tool)
- **Best for**: Let optimization choose between providers and web search methods

### `task_specific.yaml`
- **Use case**: Task-specific model variants
- **Parameters**: Customized ranges for different use cases
- **Best for**: Complex projects with different LLM needs

## Configuration Structure

Each YAML file defines:

```yaml
providers:
  openai:
    default_model: "gpt-4o-mini"
    available_models: [...]
    pricing_usd_per_1m_tokens: {...}
    model_capabilities:
      gpt-4o-mini:
        # Parameters that will be optimized
        supported_parameters: [temperature, top_p, max_tokens]
        
        # Optimization ranges
        parameter_ranges:
          temperature: [0.0, 2.0]
          max_tokens: [50, 4000]
        
        # Default values (before optimization)
        default_parameters:
          temperature: 0.7
          max_tokens: 1000
        
        # Fixed values (never optimized)
        forced_parameters: {}
```

## Customization Examples

### Conservative Classification
```yaml
gpt-4o-mini-classifier:
  supported_parameters: [temperature, max_tokens]
  parameter_ranges:
    temperature: [0.1, 0.3]  # Very deterministic
    max_tokens: [5, 50]      # Short responses
```

### Creative Writing
```yaml
gpt-4o-creative:
  supported_parameters: [temperature, top_p, frequency_penalty]
  parameter_ranges:
    temperature: [0.8, 1.5]     # High creativity
    top_p: [0.9, 1.0]          # High randomness
    frequency_penalty: [0.0, 1.0]  # Avoid repetition
```

### Fixed Parameters
```yaml
gpt-4o-factual:
  supported_parameters: [max_tokens]  # Only optimize length
  parameter_ranges:
    max_tokens: [50, 300]
  forced_parameters:
    temperature: 0.1  # Always deterministic
```

## Engineering Usage

### Basic Usage
```python
# Copy template and customize
analyzer = SentimentAnalyzer("my_sentiment_config.yaml")
```

### Production Usage
```python
# Different configs for different environments
if env == "production":
    config_file = "production_llm_config.yaml"
else:
    config_file = "development_llm_config.yaml"

llm = MultiProviderTunableLLM(config_file=config_file)
```

### Multiple Components
```python
class MySystem:
    def __init__(self, config_file: str):
        # All components use the same explicit config
        self.classifier = MultiProviderTunableLLM(
            config_file=config_file,
            default_model="gpt-4o-mini-classifier"
        )
        
        self.generator = MultiProviderTunableLLM(
            config_file=config_file, 
            default_model="gpt-4o-creative"
        )
```

## Benefits

✅ **Explicit Configuration**: No hidden global state  
✅ **Version Control**: Config files are part of your project  
✅ **Environment-Specific**: Different configs per environment  
✅ **Library-Friendly**: Perfect for embedding in other projects  
✅ **Parameter Control**: Explicitly define what gets optimized  
✅ **Provider-Aware**: Only use parameters each provider supports  

## API Key Management

Set environment variables for each provider:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Google Gemini  
export GOOGLE_API_KEY=AI...

# Anthropic (if using Claude)
export ANTHROPIC_API_KEY=sk-ant-...
```

The library will automatically detect these environment variables.
