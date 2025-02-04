# Contributing to American Option Pricing with RL

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e .
pip install -r requirements.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use several tools to maintain code quality:

- `black` for code formatting
- `flake8` for style guide enforcement
- `mypy` for static type checking
- `isort` for import sorting

Run all checks with:
```bash
pre-commit run --all-files
```

## Testing

We use pytest for testing. Run the test suite with:
```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
.
├── config/             # Configuration files
├── data/              # Data files and datasets
├── docs/              # Documentation
├── examples/          # Example notebooks
├── scripts/           # Utility scripts
├── src/               # Source code
│   ├── agents/        # RL agents
│   ├── environments/  # Option pricing environments
│   └── utils/         # Utility functions
└── tests/             # Test suite
```

## Documentation

- Use Google-style docstrings
- Update API documentation when changing interfaces
- Include examples in docstrings where appropriate

Example:
```python
def function(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Examples:
        >>> function(1, "test")
        True
    """
    pass
```

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update the docs/ with any new documentation
3. Update the examples/ if needed
4. The PR will be merged once you have the sign-off of two other developers

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/repo/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/repo/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 