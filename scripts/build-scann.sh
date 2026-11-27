#!/usr/bin/env bash
# Standalone distribution of Google's ScaNN core, with MemoryH's C-ABI bridge.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
deps=${1:-"$repo/target/scann-eval/deps"}
revision=b711c0de15aae7a27281f290e0bd76f4edf028fe
mkdir -p "$deps"
source_dir="$deps/scann-core"
if [[ ! -d "$source_dir/.git" ]]; then
  git init -q "$source_dir"
fi
if ! git -C "$source_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
  git -C "$source_dir" fetch --depth=1 https://github.com/TheCleaners/scann-core.git "$revision"
fi
git -C "$source_dir" checkout --quiet --detach "$revision"
cmake=(uv run --no-project --with cmake==4.4.5 --with ninja cmake)
"${cmake[@]}" -S "$source_dir" -B "$deps/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSCANN_BUILD_SHARED=ON -DSCANN_BUILD_STATIC=OFF \
  -DSCANN_BUILD_PYTHON=OFF -DSCANN_BUILD_RUST_BINDINGS=OFF -DSCANN_BUILD_EXAMPLES=OFF \
  -DSCANN_BUILD_TESTS=OFF -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="$repo/native/scann/bridge.cmake"
"${cmake[@]}" --build "$deps/build" --target scann_core_shared --parallel "${MEMORYH_BUILD_JOBS:-6}"
