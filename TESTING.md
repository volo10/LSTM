# Testing Guide

This document describes how to run and understand the tests for the LSTM Frequency Extraction System.

## Quick Start

### Install Testing Dependencies

```bash
pip install -r requirements.txt
```

This will install pytest and pytest-cov along with other dependencies.

### Run All Tests

```bash
# Basic test run
pytest

# Verbose output
pytest -v

# With coverage report
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file
pytest test_units.py -v
```

---

## Test Organization

### Test Files

1. **`test_units.py`** - Comprehensive unit tests
   - Tests for data generation functions
   - Tests for model architecture
   - Tests for dataset class
   - Integration tests
   - Edge case tests

2. **`test_system.py`** - System integration test
   - End-to-end workflow validation
   - Quick smoke test before training

---

## Test Categories

### 1. Data Generation Tests

**`TestSinusoidGeneration`**
- `test_generate_sinusoid_shape()` - Validates output shape
- `test_generate_sinusoid_frequency()` - Verifies frequency using FFT
- `test_generate_sinusoid_amplitude()` - Checks amplitude correctness
- `test_generate_sinusoid_phase()` - Tests phase shift behavior
- `test_generate_sinusoid_formula()` - Validates against mathematical formula

**`TestNoisySignalsGeneration`**
- `test_generate_noisy_signals_shape()` - Checks output dimensions
- `test_generate_noisy_signals_noise()` - Verifies noise addition
- `test_generate_noisy_signals_mixing()` - Tests signal mixing
- `test_generate_noisy_signals_reproducibility()` - Validates seed behavior

### 2. Dataset Tests

**`TestFrequencyExtractionDataset`**
- `test_dataset_length()` - Validates total sample count
- `test_dataset_getitem_shape()` - Checks sample dimensions
- `test_dataset_getitem_onehot()` - Tests one-hot encoding
- `test_dataset_getitem_target()` - Validates target values
- `test_dataset_instance_id_mapping()` - Tests instance tracking
- `test_dataset_all_frequencies_covered()` - Ensures all frequencies present

**`TestCreateDatasets`**
- `test_create_datasets_return_types()` - Validates function outputs
- `test_create_datasets_same_signal()` - Confirms shared signal base
- `test_create_datasets_different_noise()` - Verifies distinct noise
- `test_create_datasets_size()` - Checks dataset dimensions

### 3. Model Tests

**`TestFrequencyExtractorLSTM`**
- `test_model_initialization()` - Tests model creation
- `test_model_parameter_count()` - Validates number of parameters
- `test_model_forward_shape()` - Checks output dimensions
- `test_model_forward_no_nan()` - Ensures numerical stability
- `test_model_reset_hidden_state()` - Tests state reset
- `test_model_state_persistence()` - Validates state continuity
- `test_model_state_reset_in_forward()` - Tests reset flag
- `test_model_different_batch_sizes()` - Tests various batch sizes
- `test_model_gradient_flow()` - Validates backpropagation

### 4. Integration Tests

**`TestIntegration`**
- `test_end_to_end_training_step()` - Complete training loop
- `test_end_to_end_evaluation()` - Complete evaluation loop
- `test_dataloader_compatibility()` - PyTorch DataLoader integration

### 5. Edge Cases

**`TestEdgeCases`**
- `test_zero_noise()` - Zero noise scenario
- `test_single_frequency()` - Single frequency extraction
- `test_very_short_duration()` - Minimal duration handling
- `test_model_single_sample_batch()` - Batch size 1

---

## Running Specific Tests

### Run Tests by Category

```bash
# Run only data generation tests
pytest test_units.py::TestSinusoidGeneration -v

# Run only model tests
pytest test_units.py::TestFrequencyExtractorLSTM -v

# Run only integration tests
pytest test_units.py::TestIntegration -v
```

### Run Individual Tests

```bash
# Run a specific test
pytest test_units.py::TestSinusoidGeneration::test_generate_sinusoid_frequency -v

# Run tests matching a pattern
pytest -k "sinusoid" -v
pytest -k "model" -v
```

### Run with Markers (if defined)

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v
```

---

## Code Coverage

### Generate Coverage Report

```bash
# HTML report (opens in browser)
pytest --cov=. --cov-report=html
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
# start htmlcov/index.html  # Windows

# Terminal report
pytest --cov=. --cov-report=term

# Both
pytest --cov=. --cov-report=html --cov-report=term
```

### Coverage Metrics

The test suite aims for:
- **>90% line coverage** for core modules (`data_generator.py`, `model.py`)
- **>80% branch coverage** for conditional logic
- **100% coverage** for critical paths (model forward, data loading)

---

## Test Output Examples

### Successful Test Run

```
=============================== test session starts ================================
platform darwin -- Python 3.x.x, pytest-7.x.x
collected 45 items

test_units.py::TestSinusoidGeneration::test_generate_sinusoid_shape PASSED    [ 2%]
test_units.py::TestSinusoidGeneration::test_generate_sinusoid_frequency PASSED [ 4%]
...
test_units.py::TestEdgeCases::test_model_single_sample_batch PASSED          [100%]

=============================== 45 passed in 2.34s =================================
```

### Failed Test Example

```
FAILED test_units.py::TestSinusoidGeneration::test_generate_sinusoid_frequency
AssertionError: Peak frequency 4.8 should be close to 5.0
```

---

## Continuous Integration

### Pre-commit Testing

Before committing code, run:

```bash
# Quick smoke test
python test_system.py

# Full unit test suite
pytest

# With coverage
pytest --cov=. --cov-report=term
```

### GitHub Actions (Optional)

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Debugging Failed Tests

### Increase Verbosity

```bash
# Show print statements
pytest -v -s

# Show full traceback
pytest --tb=long

# Stop at first failure
pytest -x

# Drop into debugger on failure
pytest --pdb
```

### Run Specific Failed Tests

```bash
# Re-run last failed tests
pytest --lf

# Re-run failed tests first, then others
pytest --ff
```

---

## Writing New Tests

### Test Template

```python
class TestMyFeature:
    """Tests for my new feature."""
    
    def test_basic_functionality(self):
        """Test that the feature works in the basic case."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result.shape == expected_shape
        assert np.allclose(result, expected_value)
    
    def test_edge_case(self):
        """Test edge case behavior."""
        # Test implementation
        pass
    
    @pytest.fixture
    def sample_data(self):
        """Fixture for common test data."""
        return create_sample_dataset()
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Descriptive test names** that explain what's being tested
3. **Arrange-Act-Assert pattern** for clarity
4. **Use fixtures** for common setup
5. **Test both success and failure cases**
6. **Include edge cases** (empty inputs, boundary values, etc.)

---

## Test Performance

### Current Test Suite Performance

- **Total tests**: 45+
- **Execution time**: ~2-5 seconds
- **Coverage**: >85% of core modules

### Optimization Tips

- Use small datasets for unit tests (short duration, few samples)
- Mock expensive operations when testing logic
- Use `pytest-xdist` for parallel test execution:
  ```bash
  pip install pytest-xdist
  pytest -n auto  # Use all available cores
  ```

---

## Troubleshooting

### Common Issues

1. **Import errors**
   ```bash
   # Ensure you're in the project directory
   cd /path/to/LSTM2
   
   # Or add to PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Torch/CUDA errors**
   - Tests use CPU by default
   - GPU tests are separate and optional

3. **Random failures**
   - Check if random seeds are properly set
   - Some tests use FFT which may have small numerical variations

4. **Slow tests**
   ```bash
   # Show slowest tests
   pytest --durations=10
   ```

---

## Test Maintenance

### When to Update Tests

- ✅ After adding new features
- ✅ After fixing bugs (add regression test)
- ✅ When changing model architecture
- ✅ When modifying data generation logic
- ✅ When requirements are updated

### Code Review Checklist

- [ ] All tests pass
- [ ] New features have tests
- [ ] Coverage doesn't decrease
- [ ] Tests are documented
- [ ] No skipped/xfail tests without good reason

---

## Summary

### Essential Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run fast system check
python test_system.py

# Run specific test file
pytest test_units.py -v

# Run specific test
pytest test_units.py::TestSinusoidGeneration::test_generate_sinusoid_frequency -v
```

### Test Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| `data_generator.py` | >90% |
| `model.py` | >95% |
| `train.py` | >70% |
| `evaluate.py` | >70% |
| Overall | >85% |

---

## Additional Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Testing Best Practices**: https://docs.python-guide.org/writing/tests/

For questions or issues with tests, see the project README or create an issue on GitHub.

