#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 7.1 gateway build is Linux-only" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
cmake_bin="${FORGE_CMAKE_BIN:-/mnt/frontier-forge/tooling/cmake-4.2.3/bin/cmake}"
boost_root="${FORGE_BOOST_ROOT:-/mnt/frontier-forge/tooling/boost-1.86.0}"
if [[ ! -x "${cmake_bin}" || ! -f "${boost_root}/lib/libboost_json.a" ]]; then
  echo "run bootstrap_phase7_1_toolchain.sh first" >&2
  exit 2
fi
"${cmake_bin}" -S gateway -B gateway/build/phase7-1-release \
  -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBoost_USE_STATIC_LIBS=ON -DCMAKE_PREFIX_PATH="${boost_root}" -GNinja
"${cmake_bin}" --build gateway/build/phase7-1-release --parallel
gateway/build/phase7-1-release/forge_gateway --help >/dev/null
echo "Phase 7.1 gateway build complete"
