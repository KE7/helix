# Contributing to HELIX

Thank you for your interest in contributing to HELIX! This document provides guidelines for setting up your development environment and contributing to the project.

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/KE7/helix.git
   cd helix
   ```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```

This will install all project dependencies including development tools.

## Running Tests

Run the unit test suite:
```bash
pytest tests/unit/
```

## Type Checking

HELIX uses strict type checking with mypy. Run type checks with:
```bash
mypy --strict src/helix/
```

All code must pass strict type checking before being merged.

## Code Style

- **Formatting:** We use [ruff](https://github.com/astral-sh/ruff) for code formatting and linting
- **Type hints:** All public functions and methods must include type hints
- **Docstrings:** All public classes and functions must include docstrings describing their purpose, parameters, and return values

Run ruff to check your code:
```bash
ruff check src/
ruff format src/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the code style guidelines
4. Ensure all tests pass and type checking succeeds
5. Commit your changes with clear, descriptive commit messages
6. Push to your fork and create a Pull Request
7. Wait for CI to pass - all PRs must pass automated tests and type checking before being merged

## Questions?

If you have questions or need help, please open an issue on GitHub.
