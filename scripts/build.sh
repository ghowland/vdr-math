#!/bin/bash
# Build sdist and wheel
set -e
rm -rf dist/ build/ src/*.egg-info
python -m build
echo ""
echo "Built:"
ls -la dist/

