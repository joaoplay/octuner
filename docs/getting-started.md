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

## Step 1: Copy a Config Template

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

## Step 2: Build Your Component

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

## Step 3: Optimize (Optional)

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

## What Gets Optimized?

Octuner automatically discovers and tunes:

- **Provider & Model**: OpenAI vs Gemini, GPT-4 vs GPT-3.5, etc.
- **Temperature**: Creativity vs consistency
- **Max Tokens**: Response length
- **Other parameters**: top_p, frequency_penalty, etc.

All defined in your YAML config file.

## Next Steps

- **[Quickstart Examples](usage/quickstart.md)** - See real-world examples
- **[Provider Setup](usage/providers.md)** - Configure OpenAI, Gemini, etc.
- **[Optimization Guide](usage/optimizer.md)** - Advanced optimization strategies
- **[API Reference](reference.md)** - Complete documentation

---

**Key Principle**: Octuner is library-first. You explicitly provide config files, no global defaults or hidden state.