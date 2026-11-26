#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${1:-$repo/target/ort-cpu}"
workspace="$(cd "${1:-$repo/target/ort-cpu}" && pwd)"
version=1.32.0
archive="$workspace/onnxruntime-$version.tar.gz"
source_dir="$workspace/source"
patch_file="$repo/scripts/onnxruntime-cpu-stderr.patch"
source_url="https://downloads.sourceforge.net/project/onnx-runtime.mirror/v${version}/ONNX%20Runtime%20v${version}%20source%20code.tar.gz"
source_sha256=c8752f1f7d79a624341826d0a23a9c06f3a7ee0201f1927023355264d04137e9

if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 2 --output "$archive.part" "$source_url"
  printf '%s  %s\n' "$source_sha256" "$archive.part" | shasum -a 256 --check
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' "$source_sha256" "$archive" | shasum -a 256 --check

if [[ ! -d "$source_dir" ]]; then
  staging="$(mktemp -d "$workspace/source.XXXXXX")"
  tar -xzf "$archive" --strip-components 1 -C "$staging"
  mv "$staging" "$source_dir"
fi
if ! (cd "$source_dir" && patch -p1 -R --dry-run --silent --force < "$patch_file"); then
  (cd "$source_dir" && patch -p1 --forward --silent < "$patch_file")
fi

uv run --no-project --with cmake==4.4.5 --with ninja --with packaging \
  python "$source_dir/tools/ci_build/build.py" \
  --build_dir "$workspace/build" --config Release --update --build --build_shared_lib \
  --target onnxruntime --skip_tests --compile_no_warning_as_error --no_telemetry --skip_submodule_sync \
  --parallel "${MEMORYH_BUILD_JOBS:-8}" --cmake_generator Ninja \
  --cmake_extra_defines onnxruntime_USE_COREML=OFF onnxruntime_DISABLE_DEVICE_DISCOVERY=ON \
  FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER CMAKE_SHARED_LINKER_FLAGS=-Wl,-dead_strip_dylibs

runtime="$workspace/build/Release/libonnxruntime.$version.dylib"
dependencies="$(otool -L "$runtime")"
if [[ "$dependencies" == *libsqlite3* || "$dependencies" == *Foundation.framework/* || "$dependencies" == */CoreML.framework/* ]]; then
  printf '%s\n' "$dependencies" >&2
  exit 1
fi
printf '%s\n' "$runtime"
