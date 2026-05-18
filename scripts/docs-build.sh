#!/bin/bash
# Build mkdocs documentation
set -e
python -m mkdocs build
echo ""
echo "Docs built: site/index.html"

