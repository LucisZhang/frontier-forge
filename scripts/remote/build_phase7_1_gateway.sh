#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 7.1 gateway build is Linux-only" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
cmake -S gateway -B gateway/build/phase7-1-release \
  -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build gateway/build/phase7-1-release --parallel
gateway/build/phase7-1-release/forge_gateway --help >/dev/null
echo "Phase 7.1 gateway build complete"
