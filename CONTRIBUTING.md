# Contributing to MCP Telegram

Thank you for your interest in contributing to MCP Telegram! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the [Issues](https://github.com/TeaGuild/mcp-telegram/issues)
2. If not, create a new issue using the bug report template
3. Provide as much detail as possible to help us reproduce and fix the issue

### Suggesting Features

1. Check if the feature has already been suggested in the [Issues](https://github.com/TeaGuild/mcp-telegram/issues)
2. If not, create a new issue using the feature request template
3. Provide a clear description of the feature and its benefits

### Pull Requests

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes, following the code style guidelines
4. Write or update tests as needed
5. Update documentation to reflect your changes
6. Submit a pull request

## Development Setup

1. Clone your fork of the repository
   ```bash
   git clone https://github.com/YOUR_USERNAME/mcp-telegram.git
   cd mcp-telegram
   ```

2. Install dependencies
   ```bash
   pip install -e .
   pip install -r dev-requirements.txt
   ```

3. Set up pre-commit hooks
   ```bash
   pre-commit install
   ```

## Code Style

This project follows these code style guidelines:

- Use [Black](https://github.com/psf/black) for code formatting (line length 100)
- Use [Ruff](https://github.com/charliermarsh/ruff) for linting
- Write docstrings for all functions, classes, and modules

## Testing

- Write tests for all new features and bug fixes
- Run tests with pytest:
  ```bash
  pytest
  ```

## Documentation

- Update documentation when changing functionality
- Use clear and concise language
- Include examples where applicable

## Versioning

This project follows [Semantic Versioning](https://semver.org/).

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).