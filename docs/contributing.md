# Contributing to Octuner

Thank you for your interest in contributing to Octuner! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/octuner.git
cd octuner
```

### 2. Set Up Development Environment

Octuner uses `uv` for dependency management. If you don't have it installed:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
# Install all dependencies including dev dependencies
uv sync
```

### 3. Create a Branch

Create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:

- `feature/add-anthropic-provider` for new features
- `fix/temperature-validation` for bug fixes
- `docs/improve-quickstart` for documentation

## Making Changes

### Running Tests

Octuner includes a convenient test runner script that makes it easy to run different types of tests. Always run tests before submitting a PR!

#### Basic Usage

```bash
# Check test environment is properly set up
python run_tests.py check

# Run all tests
python run_tests.py all

# Run only unit tests
python run_tests.py unit

# Run only integration tests
python run_tests.py integration

# Run fast tests (excludes slow/external tests)
python run_tests.py fast
```

#### Advanced Options

```bash
# Run tests with coverage report
python run_tests.py unit --coverage
python run_tests.py all --coverage

# Run tests with verbose output
python run_tests.py all --verbose

# Run a specific test file or function
python run_tests.py specific --test-path tests/unit/test_providers/test_openai.py
python run_tests.py specific --test-path tests/unit/test_providers/test_openai.py::TestOpenAIProvider::test_call
```

### Adding Tests

- Add unit tests for new functionality in `tests/unit/`
- Add integration tests for end-to-end scenarios in `tests/integration/`

## Submitting a Pull Request

### 1. Commit Your Changes

Write clear, concise commit messages:

```bash
git add .
git commit -m "Add Anthropic provider support"
```

Good commit message examples:
- `Add support for Anthropic Claude models`
- `Fix temperature parameter validation`
- `Update getting started guide with new examples`

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request

1. Go to the [Octuner repository](https://github.com/joaoplay/octuner)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template with:
   - **Description**: What does this PR do?
   - **Changes**: List of main changes
   - **Testing**: How did you test this?
   - **Related Issues**: Link any related issues

## Development Guidelines

### Adding a New Provider

To add a new LLM provider:

1. Create a new file in `octuner/providers/` (e.g., `anthropic.py`)
2. Inherit from `BaseLLMProvider`
3. Implement required methods: `call()`, `configure()`, etc.
4. Register the provider in `octuner/providers/registry.py`
5. Add tests in `tests/unit/test_providers/`
6. Add a config template in `config_templates/`

Example structure:

```python
from octuner.providers.base import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    def call(self, prompt: str, **kwargs):
        # Implementation
        pass
```

### Documentation

- Update relevant documentation in `docs/`
- Add docstrings to all public functions and classes
- Follow Google-style docstring format
- Update `docs/reference.md` if you add new public APIs

### Configuration Files

- Test with different config templates
- Ensure backward compatibility
- Document any new configuration options

## Questions?

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Pull Requests**: Check existing PRs to avoid duplication

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help make Octuner better for everyone

---

Thank you for contributing to Octuner! 🎉

