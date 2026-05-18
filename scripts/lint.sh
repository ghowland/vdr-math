#!/bin/bash
# Lint with ruff, type check with mypy
set -e
echo "=== Ruff ==="
python -m ruff check src/vdr/
echo ""
echo "=== Mypy ==="
python -m mypy src/vdr/ --ignore-missing-imports

