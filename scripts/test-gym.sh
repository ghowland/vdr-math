#!/bin/bash
# Run gym integration tests
set -e
python -m pytest tests/gym/ -v "$@"

