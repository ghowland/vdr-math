#!/bin/bash
# scripts/docs-init.sh — Create docs directory structure
set -e

mkdir -p docs/api/core
mkdir -p docs/api/math
mkdir -p docs/api/signal
mkdir -p docs/api/physics
mkdir -p docs/api/ml
mkdir -p docs/api/diffusion

# Home page (copy README)
cp README.md docs/index.md

# Prose docs (copy from your documentation files)
# These should be the files we wrote: understanding doc, cookbook, migration+extension guide
touch docs/understanding.md
touch docs/cookbook.md
touch docs/migration.md
touch docs/extension.md

# Core API stubs — each file just has the mkdocstrings directive
for mod in core active fn linalg export basis; do
cat > "docs/api/${mod}.md" << EOF
# vdr.${mod}

::: vdr.${mod}
EOF
done

# Math domain stubs
for mod in number_theory continued_fractions combinatorics sequences polynomial symbolic probability geometry optimization differential_eq graph game_theory cryptographic coding_theory topology tropical control wavelets chaos transcendental; do
cat > "docs/api/math/${mod}.md" << EOF
# vdr.math.${mod}

::: vdr.math.${mod}
EOF
done

# Signal stubs
for mod in convolution dft filters schedule; do
cat > "docs/api/signal/${mod}.md" << EOF
# vdr.signal.${mod}

::: vdr.signal.${mod}
EOF
done

# Physics stubs
for mod in qed quantum orbital optics structural thermo crystallography geodesy; do
cat > "docs/api/physics/${mod}.md" << EOF
# vdr.physics.${mod}

::: vdr.physics.${mod}
EOF
done

# ML stubs
for mod in softmax exp logarithm nn autodiff optim attention transformer sampling trainer losses datasets checkpoint metrics rng init tensor; do
cat > "docs/api/ml/${mod}.md" << EOF
# vdr.ml.${mod}

::: vdr.ml.${mod}
EOF
done

# Diffusion stubs
for mod in schedule forward reverse sampling; do
cat > "docs/api/diffusion/${mod}.md" << EOF
# vdr.diffusion.${mod}

::: vdr.diffusion.${mod}
EOF
done

echo "Docs structure created. $(find docs -type f | wc -l) files."

