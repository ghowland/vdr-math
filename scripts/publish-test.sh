#!/bin/bash
# Upload to Test PyPI (test before real publish)
set -e
./scripts/build.sh
python3.12 -m twine upload --repository testpypi dist/*
echo ""
echo "Published to Test PyPI."
echo "Install with: pip install --index-url https://test.pypi.org/simple/ vdr-math"

