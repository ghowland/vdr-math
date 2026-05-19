#!/bin/bash
# Upload to real PyPI
set -e
read -p "Publishing to REAL PyPI. Are you sure? (y/N) " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 1
fi
./scripts/build.sh
python3.12 -m twine upload dist/*
echo ""
echo "Published to PyPI."
echo "Install with: pip install vdr-math"

