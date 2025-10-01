# Installation

## Requirements

- **Python 3.10+**
- **pip** or **uv**

## 1. Install

```bash
# pip
pip install octuner

# or uv (recommended)
uv add octuner
```

## 2. Configure provider credentials

Set the API keys for the providers you plan to use:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## 3. Create an explicit YAML config

Octuner is library-first and uses explicit configs only (no global defaults). Copy a starter template and keep it in your repo:

```bash
mkdir -p configs
cp config_templates/multi_provider.yaml configs/llm.yaml   # or openai_basic.yaml / gemini_basic.yaml
```

Edit `configs/llm.yaml` to include the providers/models you want enabled.

## 4. Use in your application

See Adapt existing code in Getting Started for concise before/after examples and a service wrapper pattern.

- Getting Started → Adapt existing code
- Then return to Optimization to tune parameters

## Verify

```python
import octuner
print("Octuner version:", octuner.__version__)

from octuner import MultiProviderTunableLLM
_ = MultiProviderTunableLLM(config_file="configs/llm.yaml")
print("Config loaded OK")
```

## Next steps

- Read the short [Getting Started](getting-started.md) guide
- Explore the [API Reference](reference.md) for detailed documentation
- Learn how to [contribute](contributing.md) to Octuner

Note: Always pass your YAML `config_file` explicitly when constructing `MultiProviderTunableLLM`.
