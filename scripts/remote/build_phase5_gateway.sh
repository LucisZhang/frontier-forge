#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "profile" && "$1" != "release" ) ]]; then
  echo "usage: $0 profile|release" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 5 remote gateway build is Linux-only" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
mode="$1"
build_dir="gateway/build/remote-${mode}"
common=(
  -S gateway
  -B "${build_dir}"
  -DBUILD_TESTING=OFF
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
)
if [[ -n "${FORGE_BOOST_ROOT:-}" ]]; then
  common+=("-DCMAKE_PREFIX_PATH=${FORGE_BOOST_ROOT}")
fi
if [[ "${mode}" == "profile" ]]; then
  common+=(
    "-DCMAKE_CXX_FLAGS=-pg"
    "-DCMAKE_EXE_LINKER_FLAGS=-pg"
  )
fi

cmake "${common[@]}"
cmake --build "${build_dir}" --parallel
"${build_dir}/forge_gateway" --help >/dev/null
echo "Phase 5 ${mode} gateway build complete: ${build_dir}/forge_gateway"
