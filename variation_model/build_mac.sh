#!/bin/bash
# Build VariationModelInspector.app for macOS
# Usage: bash build_mac.sh

set -e
cd "$(dirname "$0")"

PYTHON="/opt/anaconda3/envs/cv-detect/bin/python"

echo "=== Building Variation Model Inspector (macOS) ==="
echo "Python: $PYTHON"

"$PYTHON" -m PyInstaller build_mac.spec \
    --noconfirm \
    --distpath "$(pwd)/dist" \
    --workpath "$(pwd)/build"

echo "=== Build complete ==="
echo "Output: $(pwd)/dist/VariationModelInspector.app"
