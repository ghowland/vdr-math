#!/bin/bash
# Run tests with coverage
set -e
python -m pytest tests/ --cov=vdr --cov-report=html --cov-report=term-missing "$@"
echo ""
echo "HTML report: htmlcov/index.html"

