# API Reference

This section contains the auto-generated documentation for Octuner's classes, methods, and functions extracted from the source code docstrings.

---

## Core Components

### AutoTuner

The main entry point for auto-tuning LLM components. Orchestrates discovery, execution, and optimization.

::: octuner.optimization.auto.AutoTuner
    options:
      show_source: false
      members_order: source
      heading_level: 4

---

## Tunable LLM

### MultiProviderTunableLLM

A tunable LLM wrapper that optimizes provider, model, and parameter selection across multiple LLM providers.

::: octuner.tunable.tunable_llm.MultiProviderTunableLLM
    options:
      show_source: false
      members_order: source
      heading_level: 4

### TunableMixin

Base mixin class that makes any component tunable by the optimization system.

::: octuner.tunable.mixin.TunableMixin
    options:
      show_source: false
      members_order: source
      heading_level: 4

---

## Providers

### Base Provider

Abstract base class for LLM providers and standard response format.

::: octuner.providers.base.BaseLLMProvider
    options:
      show_source: false
      members_order: source
      heading_level: 4

::: octuner.providers.base.LLMResponse
    options:
      show_source: false
      heading_level: 4

### OpenAI Provider

OpenAI provider implementation using the OpenAI API.

::: octuner.providers.openai.OpenAIProvider
    options:
      show_source: false
      members_order: source
      heading_level: 4

### Gemini Provider

Google Gemini provider implementation using the google-generativeai SDK.

::: octuner.providers.gemini.GeminiProvider
    options:
      show_source: false
      members_order: source
      heading_level: 4

### Provider Registry

Functions for registering and retrieving LLM providers.

::: octuner.providers.registry.register_provider
    options:
      show_source: false

::: octuner.providers.registry.get_provider
    options:
      show_source: false

::: octuner.providers.registry.list_providers
    options:
      show_source: false

---

## Optimization

### LLMOptimizer

Main optimizer class that uses Optuna to find optimal parameter configurations.

::: octuner.optimization.optimizer.LLMOptimizer
    options:
      show_source: false
      members_order: source
      heading_level: 4

### OptimizationStrategy

Abstract base class for different optimization strategies (single objective, multi-objective).

::: octuner.optimization.optimizer.OptimizationStrategy
    options:
      show_source: false
      members_order: source
      heading_level: 4

### DatasetExecutor

Executes trials on datasets with parallel processing support.

::: octuner.optimization.executor.DatasetExecutor
    options:
      show_source: false
      members_order: source
      heading_level: 4

---

## Discovery

### ComponentDiscovery

Discovers tunable components in a component tree and builds search spaces.

::: octuner.discovery.discovery.ComponentDiscovery
    options:
      show_source: false
      members_order: source
      heading_level: 4

### Discovery Functions

::: octuner.discovery.discovery.build_search_space
    options:
      show_source: false

---

## Configuration

### ConfigLoader

Loads and validates YAML configuration files for LLM providers and models.

::: octuner.config.loader.ConfigLoader
    options:
      show_source: false
      members_order: source
      heading_level: 4

---

## Utilities

### Exporter Functions

Functions for saving and loading optimized parameters.

::: octuner.utils.exporter.save_parameters_to_yaml
    options:
      show_source: false

::: octuner.utils.exporter.load_parameters_from_yaml
    options:
      show_source: false

::: octuner.utils.exporter.create_metadata_summary
    options:
      show_source: false

### Parameter Setter

Apply optimized parameters to components.

::: octuner.utils.setter.set_parameters
    options:
      show_source: false

---

## Type Definitions

### Core Types

Type definitions and data classes used throughout Octuner.

::: octuner.tunable.types.ParamType
    options:
      show_source: false

::: octuner.tunable.types.Dataset
    options:
      show_source: false

::: octuner.tunable.types.MetricResult
    options:
      show_source: false

::: octuner.tunable.types.TrialResult
    options:
      show_source: false

::: octuner.tunable.types.SearchResult
    options:
      show_source: false

::: octuner.tunable.types.Constraints
    options:
      show_source: false

::: octuner.tunable.types.ScalarizationWeights
    options:
      show_source: false

