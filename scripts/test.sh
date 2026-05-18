#!/bin/bash
# Run all tests
set -e
python -m pytest tests/ -v "$@"

