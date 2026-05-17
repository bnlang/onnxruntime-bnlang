#!/usr/bin/env bash
# Local build (macOS / Linux). Mirror of build.ps1.
#
# Usage:
#   ./build.sh                  build for host triple (auto-detected)
#   ./build.sh --preset linux-x64
#   ./build.sh --clean

set -euo pipefail
cd "$(dirname "$0")"

PRESET=""
CLEAN=0
CONFIGURE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --clean)  CLEAN=1; shift ;;
        --configure) CONFIGURE=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Detect default preset from host
if [[ -z "$PRESET" ]]; then
    case "$(uname -s)" in
        Linux)  PRESET="linux-x64" ;;
        Darwin)
            if [[ "$(uname -m)" == "arm64" ]]; then PRESET="darwin-arm64"
            else                                    PRESET="darwin-x64"
            fi
            ;;
        *) echo "unsupported host" >&2; exit 2 ;;
    esac
fi

BUILD_DIR="build/$PRESET"
DEPS_DIR="deps/$PRESET"

if [[ "$CLEAN" -eq 1 ]]; then
    rm -rf "$BUILD_DIR"
    echo "cleaned $BUILD_DIR"
    exit 0
fi

if [[ ! -d "$DEPS_DIR" ]]; then
    echo "deps/$PRESET/ not found. Run 'bnl script/install.bnl' first." >&2
    exit 2
fi

if [[ "$CONFIGURE" -eq 1 || ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "configuring preset: $PRESET"
    cmake --preset "$PRESET"
fi

echo "building: $PRESET"
cmake --build "$BUILD_DIR" --config Release

echo "built: $BUILD_DIR"
ls -lh "$BUILD_DIR"/*.{so,dylib} 2>/dev/null || true
