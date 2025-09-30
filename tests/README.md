# Octuner Test Suite

This directory contains a comprehensive test suite for the Octuner library, covering unit tests, integration tests, and end-to-end workflows.

## 📁 Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest configuration
├── fixtures/                     # Test data files
│   ├── config_samples/           # Sample YAML configurations
│   ├── datasets/                 # Test datasets  
│   └── responses/                # Mock API responses
├── unit/                         # Unit tests
│   ├── test_config/              # Configuration loader tests
│   ├── test_tunable/             # Tunable system tests
│   ├── test_providers/           # Provider implementation tests
│   ├── test_optimization/        # Optimization algorithm tests
│   └── test_utils/               # Utility function tests
└── integration/                  # Integration tests
    └── test_end_to_end.py        # Complete workflow tests
```

## 🚀 Running Tests

### Using the Test Runner Script

```bash
# Check test environment
python run_tests.py check

# Run all unit tests
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run all tests
python run_tests.py all

# Run only fast tests (exclude slow/external)
python run_tests.py fast

# Run specific test
python run_tests.py specific --test-path tests/unit/test_config/test_loader.py

# Run with coverage
python run_tests.py unit --coverage

# Verbose output
python run_tests.py all --verbose
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/ -m unit

# Run integration tests only
pytest tests/integration/ -m integration

# Run with coverage
pytest --cov=octuner --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_config/test_loader.py

# Run specific test method
pytest tests/unit/test_config/test_loader.py::TestConfigLoader::test_init_with_valid_config

# Run tests matching pattern
pytest -k "test_config"

# Run fast tests only (exclude slow markers)
pytest -m "not slow"
```

## 🧪 Test Categories

### Unit Tests

**Configuration Tests** (`test_config/`)
- ✅ YAML configuration loading and validation
- ✅ Provider configuration parsing
- ✅ Parameter range extraction
- ✅ Error handling for malformed configs

**Tunable System Tests** (`test_tunable/`)
- ✅ TunableMixin functionality
- ✅ Parameter registration and discovery
- ✅ Registry operations
- ✅ Type definitions and data classes

**Provider Tests** (`test_providers/`)
- ✅ BaseLLMProvider interface compliance
- ✅ OpenAI provider with mocked API calls
- ✅ Gemini provider with mocked API calls
- ✅ Provider registry functionality

**Optimization Tests** (`test_optimization/`)
- ✅ DatasetExecutor trial execution
- ✅ AutoTuner optimization workflows
- ✅ Parameter search strategies
- ✅ Constraint and objective handling

**Utility Tests** (`test_utils/`)
- ✅ Parameter export functionality
- ✅ YAML file operations
- ✅ Dataset fingerprinting
- ✅ Metadata summarization

### Integration Tests

**End-to-End Workflows** (`test_end_to_end.py`)
- ✅ Complete configuration to optimization workflow
- ✅ Multi-provider setup and validation
- ✅ Custom tunable component integration
- ✅ Error handling throughout the pipeline
- ✅ Performance characteristics

## 🎯 Testing Strategy

### Mocking Strategy

**External APIs**: All OpenAI and Gemini API calls are mocked to ensure:
- Tests run without API keys
- Predictable, deterministic results
- Fast execution
- No external dependencies

**File System**: Temporary directories used for:
- Configuration file testing
- Result export verification
- No permanent files created

**Optimization**: Optuna trials mocked for:
- Predictable optimization results
- Fast test execution
- Controlled parameter exploration

### Test Data

**Fixtures**: Comprehensive fixtures provide:
- Sample configurations for all providers
- Test datasets for different domains
- Mock API responses
- Temporary file management

**Realistic Scenarios**: Tests include:
- Valid and invalid configurations
- Successful and failed API calls
- Different optimization modes
- Edge cases and error conditions

## 📊 Coverage Goals

- **Unit Tests**: >90% line coverage
- **Integration Tests**: Cover main user workflows
- **Critical Paths**: 100% coverage on error handling

## 🔧 Test Configuration

### pytest.ini Configuration

```ini
[tool:pytest]
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests  
    slow: Slow running tests
    external: Tests requiring external services
```

### Environment Variables

Tests use mocked environment variables:
- `OPENAI_API_KEY`: test-openai-key
- `GOOGLE_API_KEY`: test-google-key
- `GOOGLE_SEARCH_API_KEY`: test-search-key
- `GOOGLE_SEARCH_ENGINE_ID`: test-engine-id

## 🚨 Common Issues and Solutions

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'octuner'`

**Solution**: Install the package in development mode:
```bash
pip install -e .
```

### API Key Warnings

**Problem**: Warnings about missing API keys

**Solution**: Tests use mocked environment variables automatically. Real API keys are not needed.

### Slow Tests

**Problem**: Tests running slowly

**Solution**: Run fast tests only:
```bash
python run_tests.py fast
```

### Coverage Reports

**Problem**: No coverage reports generated

**Solution**: Install pytest-cov:
```bash
pip install pytest-cov
```

## 🔍 Debugging Tests

### Verbose Output

```bash
# Show detailed test output
pytest -v

# Show print statements
pytest -s

# Show local variables on failure
pytest --tb=long

# Drop into debugger on failure
pytest --pdb
```

### Specific Test Debugging

```bash
# Run single test with debugging
pytest tests/unit/test_config/test_loader.py::TestConfigLoader::test_init_with_valid_config -v -s

# Run with coverage for specific module
pytest tests/unit/test_config/ --cov=octuner.config --cov-report=term-missing
```

## 📈 Test Metrics

Current test coverage:
- **Configuration Module**: ~95% coverage
- **Tunable System**: ~90% coverage  
- **Provider Implementations**: ~85% coverage
- **Optimization Engine**: ~90% coverage
- **Utilities**: ~95% coverage

## 🤝 Contributing Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Follow naming conventions**: `test_feature_description`
3. **Use appropriate fixtures** from `conftest.py`
4. **Mock external dependencies** (APIs, file system)
5. **Test both success and failure cases**
6. **Add docstrings** explaining test purpose
7. **Keep tests focused** and independent

### Example Test Structure

```python
def test_feature_description(self, fixture1, fixture2):
    """Test that feature works correctly under normal conditions."""
    # Arrange
    component = create_test_component(fixture1)
    
    # Act
    result = component.perform_action(fixture2)
    
    # Assert
    assert result.is_successful()
    assert result.value == expected_value
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Python Testing Best Practices](https://realpython.com/python-testing/)

---

For questions about the test suite, please refer to the main README or open an issue on the project repository.
