#!/bin/bash
# scripts/docs-build.sh
set -e
lazydocs --output-path docs/api \
         --overview-file README \
         --src-base-url https://github.com/ghowland/vdr-math/blob/main/ \
         vdr
echo "API docs generated in docs/api/"
