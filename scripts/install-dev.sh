#!/bin/bash
# Set up development environment
set -e
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
echo ""
echo "Dev environment ready. Run ./scripts/test.sh to verify."

