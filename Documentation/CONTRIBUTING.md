# Contributing Guide

Thank you for your interest in contributing to the LSTM Frequency Extraction System!

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Code Review](#code-review)

---

## Getting Started

### Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add tests
- 🔧 Fix issues
- ✨ Implement new features
- 🎨 Improve visualizations
- ⚡ Optimize performance

### Before Contributing

1. **Check existing issues:** https://github.com/volo10/LSTM/issues
2. **Read documentation:** Especially ARCHITECTURE.md and API_DOCUMENTATION.md
3. **Run tests:** Ensure current tests pass
4. **Discuss major changes:** Open an issue first for large features

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork:
git clone https://github.com/YOUR_USERNAME/LSTM.git
cd LSTM

# Add upstream remote:
git remote add upstream https://github.com/volo10/LSTM.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python3 -m venv lstm_env
source lstm_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install black flake8 mypy pylint
```

### 3. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/bug-description
```

### 4. Verify Setup

```bash
# Run tests
pytest test_units.py -v

# Run system test
python3 test_system.py
```

---

## Contribution Workflow

### Standard Workflow

```bash
# 1. Create branch
git checkout -b feature/new-feature

# 2. Make changes
# ... edit files ...

# 3. Test changes
pytest test_units.py -v
python3 test_system.py

# 4. Commit changes
git add .
git commit -m "Add new feature: description"

# 5. Push to your fork
git push origin feature/new-feature

# 6. Create Pull Request on GitHub
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Update your main branch
git checkout main
git merge upstream/main

# Update your feature branch
git checkout feature/your-feature
git rebase main
```

---

## Coding Standards

### Python Style Guide

Follow PEP 8 with these specific guidelines:

**1. Code Formatting:**
```python
# Use 4 spaces for indentation (not tabs)
# Maximum line length: 88 characters (Black default)
# Use double quotes for strings

def example_function(param1: int, param2: str) -> float:
    """
    Brief description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
    result = param1 * len(param2)
    return float(result)
```

**2. Naming Conventions:**
```python
# Functions and variables: snake_case
def calculate_frequency():
    sample_rate = 1000

# Classes: PascalCase
class FrequencyExtractor:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001

# Private methods/variables: _leading_underscore
def _internal_helper():
    pass
```

**3. Type Hints:**
```python
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch

def process_signal(
    signal: np.ndarray,
    frequencies: List[float],
    device: str = 'cpu'
) -> torch.Tensor:
    """Always use type hints for function signatures."""
    pass
```

**4. Docstrings:**
```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int
) -> Dict[str, List[float]]:
    """
    Train the LSTM model.
    
    This function trains the model for the specified number of epochs,
    tracking training and validation loss.
    
    Args:
        model: The LSTM model to train
        dataloader: DataLoader for training data
        epochs: Number of training epochs
    
    Returns:
        Dictionary containing training history:
        - 'train_loss': List of training losses per epoch
        - 'val_loss': List of validation losses per epoch
    
    Example:
        >>> model = FrequencyExtractorLSTM()
        >>> history = train_model(model, train_loader, 100)
        >>> print(history['train_loss'][0])
        0.452
    """
    pass
```

### Code Quality Tools

**Black (Code Formatter):**
```bash
# Install
pip install black

# Format code
black .

# Check without modifying
black --check .
```

**Flake8 (Linter):**
```bash
# Install
pip install flake8

# Run linter
flake8 .

# Configuration in .flake8 or setup.cfg:
# [flake8]
# max-line-length = 88
# extend-ignore = E203, W503
```

**MyPy (Type Checker):**
```bash
# Install
pip install mypy

# Run type checker
mypy *.py
```

---

## Testing Guidelines

### Writing Tests

**1. Test Structure:**
```python
import pytest
import torch
from data_generator import generate_single_sinusoid

class TestSinusoidGeneration:
    """Tests for sinusoid generation."""
    
    def test_correct_shape(self):
        """Test that output has correct shape."""
        # Arrange
        frequency = 5.0
        sampling_rate = 1000
        duration = 1.0
        
        # Act
        signal = generate_single_sinusoid(
            frequency=frequency,
            amplitude=1.0,
            phase=0.0,
            sampling_rate=sampling_rate,
            duration=duration
        )
        
        # Assert
        expected_length = int(sampling_rate * duration)
        assert len(signal) == expected_length
```

**2. Test Coverage:**
- Aim for >80% code coverage
- Test normal cases
- Test edge cases
- Test error conditions

**3. Running Tests:**
```bash
# Run all tests
pytest

# Run specific test file
pytest test_units.py

# Run specific test class
pytest test_units.py::TestSinusoidGeneration

# Run specific test
pytest test_units.py::TestSinusoidGeneration::test_correct_shape

# With coverage
pytest --cov=. --cov-report=html

# Verbose output
pytest -v -s
```

**4. Test Requirements:**
- All new features must have tests
- Bug fixes should include regression tests
- Tests must pass before PR can be merged
- Add docstrings to test functions

### Testing Checklist

Before submitting PR:
- [ ] All existing tests pass
- [ ] New tests added for new features
- [ ] Edge cases covered
- [ ] No decrease in code coverage
- [ ] Tests are documented

---

## Documentation

### Documentation Standards

**1. Code Comments:**
```python
# Good: Explain WHY, not WHAT
# Reset state to prevent gradient flow between signal instances
model.reset_hidden_state(batch_size=1, device='cpu')

# Bad: Obvious comments
# Set x to 5
x = 5
```

**2. Function Documentation:**
- Always include docstring
- Describe parameters and return values
- Include usage examples for complex functions
- Document any side effects

**3. Module Documentation:**
```python
"""
Data Generator for LSTM Frequency Extraction

This module provides functions for generating synthetic signals
and creating PyTorch datasets for training.

Main components:
- generate_single_sinusoid: Create a single sinusoidal signal
- generate_noisy_signals: Create mixed noisy signals
- FrequencyExtractionDataset: PyTorch Dataset class
"""
```

**4. Updating Documentation:**

When adding features, update:
- [ ] Function/class docstrings
- [ ] API_DOCUMENTATION.md (if public API changes)
- [ ] ARCHITECTURE.md (if architecture changes)
- [ ] README.md (if user-facing changes)
- [ ] CHANGELOG.md (add entry)

---

## Pull Request Process

### Before Submitting PR

**Checklist:**
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No unnecessary files included
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] System test passes
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #123
```

### PR Review Process

1. **Automated Checks:**
   - Tests must pass
   - Code coverage maintained

2. **Code Review:**
   - At least one approval required
   - Address reviewer comments

3. **Merge:**
   - Squash commits if many small commits
   - Use descriptive merge commit message

---

## Issue Guidelines

### Reporting Bugs

Use this template:

```markdown
## Bug Description
Clear description of the bug

## To Reproduce
1. Step 1
2. Step 2
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: macOS 13.0
- Python: 3.10.2
- PyTorch: 2.0.0
- Commit: abc1234

## Error Message
```
Full error traceback here
```

## Additional Context
Any other relevant information
```

### Suggesting Features

Use this template:

```markdown
## Feature Description
Clear description of proposed feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches you considered

## Additional Context
Examples, mockups, references
```

---

## Code Review

### As a Reviewer

**What to Look For:**
1. **Correctness:** Does the code do what it's supposed to?
2. **Tests:** Are there adequate tests?
3. **Style:** Does it follow coding standards?
4. **Documentation:** Is it well-documented?
5. **Performance:** Are there obvious inefficiencies?
6. **Security:** Any security concerns?

**Review Etiquette:**
- Be respectful and constructive
- Explain why changes are needed
- Suggest specific improvements
- Acknowledge good work

**Example Comments:**
```
✅ Good: "Consider using torch.no_grad() here to reduce memory usage"
❌ Bad: "This is wrong"

✅ Good: "This looks great! Just one minor suggestion about the docstring"
❌ Bad: "LGTM"
```

### As a Contributor

**Responding to Reviews:**
- Address all comments
- Ask for clarification if needed
- Don't take feedback personally
- Update PR based on feedback
- Mark conversations as resolved when fixed

---

## Development Tips

### Debugging

```python
# Print tensor shapes
print(f"Input: {x.shape}, Output: {y.shape}")

# Check for NaN/Inf
assert not torch.isnan(x).any(), "Input contains NaN"
assert not torch.isinf(x).any(), "Input contains Inf"

# Profile performance
import time
start = time.time()
model(x)
print(f"Forward pass: {time.time() - start:.4f}s")

# Use logging instead of print
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Processing batch {i}")
```

### Common Tasks

**Add New Frequency:**
```python
# 1. Update config.yaml
frequencies: [1.0, 3.0, 5.0, 7.0, 9.0]  # Add 9.0

# 2. Update model input size if needed
input_size = 1 + len(frequencies)  # Now 6

# 3. Update tests
# 4. Retrain model
# 5. Update documentation
```

**Add New Model Architecture:**
```python
# 1. Create new model class in model.py
class FrequencyExtractorGRU(nn.Module):
    # ... implementation ...

# 2. Add tests in test_units.py
# 3. Update train.py to support new model
# 4. Document in API_DOCUMENTATION.md
# 5. Compare performance in TRAINING_RESULTS.md
```

**Optimize Performance:**
```python
# 1. Profile code
python3 -m cProfile -o profile.stats train.py

# 2. Analyze results
python3 -m pstats profile.stats
# > sort cumtime
# > stats 10

# 3. Optimize bottlenecks
# 4. Measure improvement
# 5. Add performance tests
```

---

## Release Process

For maintainers:

```bash
# 1. Update version
# Update in README.md, setup.py, etc.

# 2. Update CHANGELOG.md
# Document all changes since last release

# 3. Create release branch
git checkout -b release/v1.1.0

# 4. Final testing
pytest
python3 test_system.py

# 5. Merge to main
git checkout main
git merge release/v1.1.0

# 6. Tag release
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0

# 7. Create GitHub release
# Go to GitHub → Releases → Create new release
# Attach trained models if applicable
```

---

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person
- Assume good intentions

### Communication Channels

- **Issues:** Bug reports and feature requests
- **Pull Requests:** Code contributions
- **Discussions:** General questions and ideas

---

## Recognition

Contributors will be acknowledged in:
- README.md (Contributors section)
- Release notes
- Git commit history

---

## Questions?

- Check existing documentation
- Search closed issues
- Open a new issue with "Question" label
- Be specific about what you need help with

---

## Additional Resources

- **Python Style Guide:** https://pep8.org/
- **PyTorch Documentation:** https://pytorch.org/docs/
- **Git Workflow:** https://guides.github.com/
- **Testing with pytest:** https://docs.pytest.org/

---

**Thank you for contributing to the LSTM Frequency Extraction System!**

**Last Updated:** November 2025  
**Version:** 1.0

