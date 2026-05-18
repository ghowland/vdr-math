#!/bin/bash
# Run core tests (skip gym)
set -e
python -m pytest tests/ -v --ignore=tests/gym "$@"

