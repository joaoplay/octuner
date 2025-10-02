<div align="center">
  <img src="assets/octuner_logo.png" alt="Octuner Logo" width="400" style="margin-bottom: 20px;">
</div>

# Octuner - Multi-Provider LLM Optimizer

**Optimize LLM providers, models, and parameters — without the guesswork.**

Octuner is a lightweight library that solves the decision-making process when integrating with LLMs, especially in multi-step model chaining scenarios.

## Why Octuner?

Building LLM applications often feels like solving a puzzle:

- **Which provider?** OpenAI, Gemini, Anthropic… or self-hosted (Ollama, vLLM, etc.)?  
- **Which model?** GPT-4o, Gemini Pro, Claude…?  
- **Which parameters?** Temperature, top-p, max_tokens…?  
- **How to balance quality, cost, and latency?**

Things get harder with **model chaining**, where each step depends on the previous one:

```
Input → [LLM A] → Intermediate Result → [LLM B] → Final Output
```

Manual trial-and-error leads to inconsistent performance, wasted budget, and provider lock-in. Octuner removes the guesswork.

## Quick Start

Build a tiny sentiment chain that first explains why, then outputs a single-word label. You’ll pass an explicit YAML config path so it’s ready for optimization.

### 1. Create your model chain

```python
from octuner import MultiProviderTunableLLM

class SentimentChain:
    def __init__(self, config_file: str):
        # Reason step (clear explanation)
        self.reasoner = MultiProviderTunableLLM(
            config_file,
            default_provider="openai",
            default_model="gpt-4o-mini",
        )
        # Label step (concise single-word output)
        self.labeler = MultiProviderTunableLLM(
            config_file,
            default_provider="gemini",
            default_model="gemini-1.5-flash",
        )

    def _build_reason_prompt(self, text: str) -> str:
        return (
            "Explain the sentiment (positive/negative/neutral) of the text below. "
            "Keep the reasoning short and specific.\n\n"
            f"Text: {text}\n"
        )

    def _build_label_prompt(self, reasoning: str) -> str:
        return (
            "Given the reasoning below, respond with only one word: "
            "positive | negative | neutral.\n\n"
            f"Reasoning:\n{reasoning}\n"
        )

    def predict(self, text: str) -> dict:
        reason = self.reasoner.call(self._build_reason_prompt(text)).text
        label = self.labeler.call(self._build_label_prompt(reason)).text.strip().lower()
        return {"sentiment": label, "why": reason}
```

### 2. Add a dataset and metric

```python
dataset = [
    {"input": "I love this!", "target": {"sentiment": "positive"}},
    {"input": "This is awful.", "target": {"sentiment": "negative"}},
    {"input": "It's fine.", "target": {"sentiment": "neutral"}},
]

def metric(output, target):
    return 1.0 if output["sentiment"] == target["sentiment"] else 0.0
```

### 3. Optimize

```python
from octuner import AutoTuner, apply_best

chain = SentimentChain("configs/llm.yaml")  # explicit YAML config path

tuner = AutoTuner.from_component(
    component=chain,
    entrypoint=lambda c, x: c.predict(x),
    dataset=dataset,
    metric=metric,
)

# Focus on the most impactful knobs first
tuner.include([
    "reasoner.provider_model", "reasoner.temperature",
    "labeler.provider_model", "labeler.temperature",
])

result = tuner.search(max_trials=12, mode="pareto")
result.save_best("optimized_sentiment_chain.yaml")

apply_best(chain, "optimized_sentiment_chain.yaml")
print(chain.predict("The new UI is a joy to use."))
```

### Full example

```python
from octuner import MultiProviderTunableLLM, AutoTuner, apply_best

class SentimentChain:
    def __init__(self, config_file: str):
        self.reasoner = MultiProviderTunableLLM(
            config_file,
            default_provider="openai",
            default_model="gpt-4o-mini",
        )
        self.labeler = MultiProviderTunableLLM(
            config_file,
            default_provider="gemini",
            default_model="gemini-1.5-flash",
        )

    def _build_reason_prompt(self, text: str) -> str:
        return (
            "Explain the sentiment (positive/negative/neutral) of the text below. "
            "Keep the reasoning short and specific.\n\n"
            f"Text: {text}\n"
        )

    def _build_label_prompt(self, reasoning: str) -> str:
        return (
            "Given the reasoning below, respond with only one word: "
            "positive | negative | neutral.\n\n"
            f"Reasoning:\n{reasoning}\n"
        )

    def predict(self, text: str) -> dict:
        reason = self.reasoner.call(self._build_reason_prompt(text)).text
        label = self.labeler.call(self._build_label_prompt(reason)).text.strip().lower()
        return {"sentiment": label, "why": reason}

dataset = [
    {"input": "I love this!", "target": {"sentiment": "positive"}},
    {"input": "This is awful.", "target": {"sentiment": "negative"}},
    {"input": "It's fine.", "target": {"sentiment": "neutral"}},
]

def metric(output, target):
    return 1.0 if output["sentiment"] == target["sentiment"] else 0.0

chain = SentimentChain("configs/llm.yaml")

tuner = AutoTuner.from_component(
    component=chain,
    entrypoint=lambda c, x: c.predict(x),
    dataset=dataset,
    metric=metric,
)

tuner.include([
    "reasoner.provider_model", "reasoner.temperature",
    "labeler.provider_model", "labeler.temperature",
])

result = tuner.search(max_trials=12, mode="pareto")
result.save_best("optimized_sentiment_chain.yaml")

apply_best(chain, "optimized_sentiment_chain.yaml")
print(chain.predict("The new UI is a joy to use."))
```

## Key Features

### **Multi-Provider Optimization**
Automatically discover the best combination of:

- **Providers**: OpenAI, Gemini, Anthropic, or self-hosted (Ollama, vLLM, etc.)
- **Models**: GPT-4o, Gemini Pro, Claude, or self-hosted (Llama, Mistral, etc.)
- **Parameters**: temperature, top_p, max_tokens, web search
- **Capabilities**: Web search, etc.

### **Multiple Optimization Modes**
- **Pareto**: Balance quality, cost, and latency (default)
- **Constrained**: Maximize quality within cost/latency limits
- **Scalarized**: Optimize weighted combination of metrics
- **Quality-focused**: Maximize performance regardless of cost/time
- **Cost-focused**: Minimize spending while meeting quality thresholds
- **Speed-focused**: Optimize for fastest response within quality bounds

### **Flexible Parameter Control**
```yaml
providers:
  openai:
    model_capabilities:
      gpt-4o-mini:
        supported_parameters: [temperature, top_p, max_tokens]
        parameter_ranges:
          temperature: [0.0, 2.0]
          max_tokens: [50, 4000]
        default_parameters:
          temperature: 0.7
          max_tokens: 1000
```

### **Web Search Integration**
- **OpenAI**: Built-in web search capabilities
- **Gemini**: Native Google grounding tool for web context
- **Tunable**: Let optimization decide when web search improves performance

## Learn More

1. **[Installation & Setup](installation.md)** - Get started quickly
2. **[Getting Started](getting-started.md)** - Complete workflow, examples, and custom provider setup  
3. **[API Reference](reference.md)** - Complete API documentation
4. **[Contributing](contributing.md)** - How to contribute to Octuner

---

*Octuner helps developers build better LLM applications by systematically optimizing the quality vs cost vs time triangle through explicit configuration management and data-driven parameter tuning.*
