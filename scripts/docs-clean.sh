#!/bin/bash
# Clean all build artifacts
rm -rf dist/ build/ src/*.egg-info
rm -rf .pytest_cache/ htmlcov/ .coverage
rm -rf site/
rm -rf .mypy_cache/
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "Cleaned."

