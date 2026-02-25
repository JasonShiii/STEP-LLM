# Contributing to CAD Code-Based Generation

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see [Development Setup](#development-setup))
4. Create a branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## How to Contribute

### Types of Contributions

We welcome contributions in the following areas:

- **Bug fixes**: Fix issues reported in the issue tracker
- **New features**: Add new functionality or improve existing features
- **Documentation**: Improve README, docstrings, or add examples
- **Tests**: Add or improve test coverage
- **Performance**: Optimize code for better performance
- **Dataset improvements**: Enhance data processing or filtering
- **Model improvements**: Improve training procedures or architectures

### Before You Start

- Check the issue tracker to see if someone is already working on the feature/bug
- For major changes, open an issue first to discuss your approach
- Make sure your contribution aligns with the project's goals

## Development Setup

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/cad_codebased.git
cd cad_codebased

# Create conda environment
conda create -n cad_dev python=3.10
conda activate cad_dev

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy pre-commit
```

### Configuration

1. **Set up environment variables**

Create a `.env` file in the project root:

```bash
# OpenAI API (for captioning)
OPENAI_API_KEY=your-key-here

# WandB (optional, for experiment tracking)
WANDB_API_KEY=your-key-here

# Project paths (adjust to your setup)
PROJECT_ROOT=/path/to/cad_codebased
DATA_DIR=/path/to/data
```

2. **Update paths in scripts**

Many scripts contain hardcoded paths. Update these to use environment variables or relative paths.

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

### Code Formatting

We use `black` for code formatting:

```bash
# Format your code
black .

# Check formatting without making changes
black --check .
```

### Linting

Use `flake8` to check code quality:

```bash
flake8 --max-line-length=120 --ignore=E203,W503 .
```

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md if adding new features
- Add inline comments for complex logic

Example docstring:

```python
def generate_step_file(prompt: str, model_path: str, use_rag: bool = False) -> str:
    """
    Generate a STEP file from a text prompt.

    Args:
        prompt: Natural language description of the CAD model
        model_path: Path to the trained model checkpoint
        use_rag: Whether to use retrieval-augmented generation

    Returns:
        Generated STEP file content as a string

    Raises:
        ValueError: If the model path doesn't exist
        RuntimeError: If generation fails
    """
    # Implementation...
```

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Add a detailed description if needed

Example:

```
Add multi-GPU support for STEP generation

- Implement data parallelism across GPUs
- Add command-line arguments for GPU selection
- Update documentation with usage examples
```

### Pull Request Process

1. **Update your branch**

```bash
git fetch upstream
git rebase upstream/main
```

2. **Run tests**

Ensure all tests pass before submitting:

```bash
pytest tests/
```

3. **Create pull request**

- Push your branch to your fork
- Open a pull request on GitHub
- Fill out the PR template with:
  - Description of changes
  - Related issue number (if applicable)
  - Testing performed
  - Screenshots (if UI changes)

4. **Code review**

- Address reviewer comments
- Keep the PR focused on a single feature/fix
- Be responsive to feedback

5. **Merge**

Once approved, your PR will be merged by a maintainer.

## Reporting Bugs

### Before Reporting

- Check if the bug has already been reported
- Try to reproduce the bug with the latest version
- Gather relevant information (OS, Python version, dependencies)

### Bug Report Template

When reporting a bug, include:

```markdown
**Description**
A clear description of the bug.

**Steps to Reproduce**
1. Step 1
2. Step 2
3. ...

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.12]
- CUDA version: [e.g., 11.8]
- Relevant package versions

**Additional Context**
Error messages, logs, screenshots, etc.
```

## Feature Requests

We welcome feature requests! Please provide:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: Your idea for implementing the feature
- **Alternatives**: Other approaches you've considered
- **Impact**: Who would benefit from this feature

## Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Reach out to the maintainers
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions help make this project better for everyone. We appreciate your time and effort!
