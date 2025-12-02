# Contributing to Dynamical SGD

We welcome contributions to the Dynamical SGD project! This document provides guidelines for contributing to help maintain code quality and consistency.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Submitting Changes](#submitting-changes)
6. [Issue Guidelines](#issue-guidelines)
7. [Feature Requests](#feature-requests)

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/dynamical-sgd.git
   cd dynamical-sgd
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/NicolasRW/dynamical-sgd.git
   ```

### Stay Updated

Before starting work, ensure your fork is up to date:
```bash
git checkout main
git pull upstream main
git push origin main
```

## Development Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks (recommended):**
   ```bash
   pre-commit install
   ```

## Code Style

### Python Style Guidelines

- Follow [PEP 8](https://pep8.org/) for Python code style
- Use [Black](https://black.readthedocs.io/) for code formatting (max line length: 88)
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use type hints where possible

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstrings:
  ```python
  def example_function(param1: int, param2: str) -> bool:
      """
      Brief description of the function.
      
      Args:
          param1: Description of parameter 1
          param2: Description of parameter 2
      
      Returns:
          Description of return value
      
      Raises:
          ValueError: Description of when this exception is raised
      """
  ```

### Import Organization

Organize imports in the following order:
1. Standard library imports
2. Third-party imports
3. Local application imports

Use absolute imports when possible.

### JAX-Specific Guidelines

- Use `jax.jit` decorations appropriately for performance
- Prefer functional programming patterns
- Use `jax.tree_util` for working with parameter pytrees
- Include `static_argnums` in jit decorators when needed

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_spiral_classifier.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both unit tests and integration tests
- Test edge cases and error conditions

Example test structure:
```python
import pytest
import jax.numpy as jnp
from src.models.spiral_classifier import SpiralClassifier

class TestSpiralClassifier:
    def test_initialization(self):
        """Test classifier initialization with default parameters."""
        classifier = SpiralClassifier()
        assert classifier.num_classes == 3
        assert classifier.nn_width == 100
    
    def test_dataset_generation(self):
        """Test spiral dataset generation."""
        classifier = SpiralClassifier()
        X, Y = classifier.make_dataset()
        assert X.shape == (300, 2)  # 3 classes × 100 points × 2 features
        assert Y.shape == (300, 3)  # 3 classes × 100 points × 3 classes
```

## Submitting Changes

### Branch Naming

Create descriptive branch names:
- `feature/add-new-optimizer`
- `bugfix/fix-gradient-computation`
- `docs/update-installation-guide`
- `refactor/reorganize-config-system`

### Commit Messages

Write clear, descriptive commit messages:
```
Add support for RMSprop optimizer

- Implement RMSprop in optimizer configuration
- Add tests for new optimizer
- Update documentation with RMSprop usage examples
```

### Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference any related issues
   - Include test results
   - Add screenshots for UI changes

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] No conflicts with main branch
- [ ] Commit messages are descriptive

## Issue Guidelines

### Bug Reports

When reporting bugs, include:
- **Environment details**: OS, Python version, JAX version
- **Reproduction steps**: Minimal code example
- **Expected vs actual behavior**
- **Error messages** (full traceback if applicable)
- **Configuration files** (if relevant)

### Feature Requests

For feature requests, provide:
- **Clear description** of the proposed feature
- **Use case**: Why is this feature needed?
- **Proposed implementation** (if you have ideas)
- **Backwards compatibility** considerations

## Areas for Contribution

We especially welcome contributions in:

### Core Features
- New optimization algorithms
- Additional neural network architectures
- Advanced analysis methods
- Performance optimizations

### Analysis Tools
- Statistical analysis functions
- Visualization improvements
- Metric computation enhancements
- Data export utilities

### Documentation
- Tutorial notebooks
- API documentation
- Usage examples
- Scientific background explanations

### Testing
- Unit test coverage
- Integration tests
- Performance benchmarks
- Regression tests

### Infrastructure
- CI/CD improvements
- Docker containerization
- Cloud deployment scripts
- Package distribution

## Code of Conduct

This project follows a standard code of conduct:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Credit others for their contributions
- Respect different viewpoints and experiences

## Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Ask questions** in GitHub discussions
4. **Contact maintainers** via email if needed

## Recognition

Contributors will be recognized in:
- `AUTHORS.md` file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to Dynamical SGD! Your help makes this project better for everyone.