#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 7.1 gateway toolchain bootstrap is Linux-only" >&2
  exit 2
fi
: "${FORGE_BENCH_GIT_SHA:?Set FORGE_BENCH_GIT_SHA to the benchmark commit}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_real="$(readlink -f "${repo_root}")"
if [[ "${repo_real}" != /mnt/frontier-forge/* ]] || ! mountpoint -q /mnt/frontier-forge; then
  echo "repo and toolchain must live on the mounted /mnt/frontier-forge data disk" >&2
  exit 2
fi
cd "${repo_root}"
if [[ "$(git rev-parse HEAD)" != "${FORGE_BENCH_GIT_SHA}" ]]; then
  echo "checked-out commit differs from FORGE_BENCH_GIT_SHA" >&2
  exit 2
fi
if [[ ! -x .venv-phase4/bin/python ]]; then
  echo "bootstrap_phase7_1_a10.sh must create Python 3.12 first" >&2
  exit 2
fi

cache_root="${FORGE_CACHE_ROOT:-/mnt/frontier-forge/cache}"
tooling_root="${FORGE_TOOLING_ROOT:-/mnt/frontier-forge/tooling}"
cmake_version=4.2.3
cmake_sha256=8e91b381aaea3c47110583dccc52f4562333d1accdbb806939f953c16e74ec0a
cmake_venv="${tooling_root}/cmake-${cmake_version}"
boost_version=1.86.0
boost_slug=boost_1_86_0
boost_sha256=1bed88e40401b2cb7a1f76d4bab499e352fa4d0c5f31c0dbae64e24d34d7513b
boost_source_url="https://archives.boost.io/release/${boost_version}/source/${boost_slug}.tar.bz2"
boost_download_url="${FORGE_BOOST_DOWNLOAD_URL:-https://sources.cdn.immortalwrt.org/${boost_slug}.tar.bz2}"
boost_archive="${cache_root}/downloads/${boost_slug}.tar.bz2"
boost_source="${tooling_root}/src/${boost_slug}"
boost_prefix="${tooling_root}/boost-${boost_version}"

mkdir -p "${cache_root}/downloads" "${cache_root}/pip" "${cache_root}/tmp" \
  "${tooling_root}/src" results/phase7_1
export PIP_CACHE_DIR="${cache_root}/pip"
export TMPDIR="${cache_root}/tmp"

cmake_requirements=results/phase7_1/a10_cmake_toolchain_requirements.txt
printf 'cmake==%s --hash=sha256:%s\n' "${cmake_version}" "${cmake_sha256}" \
  > "${cmake_requirements}"
if [[ ! -x "${cmake_venv}/bin/cmake" ]]; then
  .venv-phase4/bin/python -m venv "${cmake_venv}"
  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=180 \
    "${cmake_venv}/bin/python" -m pip install \
      --require-hashes --no-deps --index-url https://mirrors.aliyun.com/pypi/simple \
      --progress-bar off -r "${cmake_requirements}"
fi
if [[ "$("${cmake_venv}/bin/cmake" --version | awk 'NR == 1 {print $3}')" != "${cmake_version}" ]]; then
  echo "pinned CMake version verification failed" >&2
  exit 1
fi

if [[ ! -f "${boost_archive}" ]]; then
  curl --fail --location --retry 5 --retry-delay 2 --silent --show-error \
    --output "${boost_archive}.part" "${boost_download_url}"
  if [[ "$(sha256sum "${boost_archive}.part" | awk '{print $1}')" != "${boost_sha256}" ]]; then
    echo "downloaded Boost archive hash mismatch" >&2
    exit 1
  fi
  mv "${boost_archive}.part" "${boost_archive}"
fi
if [[ "$(sha256sum "${boost_archive}" | awk '{print $1}')" != "${boost_sha256}" ]]; then
  echo "cached Boost archive hash mismatch" >&2
  exit 1
fi
if [[ ! -d "${boost_source}" ]]; then
  tar -xjf "${boost_archive}" -C "${tooling_root}/src"
fi
if [[ ! -f "${boost_prefix}/lib/cmake/Boost-${boost_version}/BoostConfig.cmake" ]]; then
  (
    cd "${boost_source}"
    ./bootstrap.sh --prefix="${boost_prefix}" --with-libraries=json,system
    ./b2 --prefix="${boost_prefix}" --with-json --with-system \
      "-j$(nproc)" variant=release link=static threading=multi \
      cxxflags=-fPIC install
  )
fi
if [[ ! -f "${boost_prefix}/lib/libboost_json.a" ]]; then
  echo "pinned Boost.JSON static library is missing" >&2
  exit 1
fi

jq -n \
  --arg git_sha "${FORGE_BENCH_GIT_SHA}" \
  --arg cmake_version "${cmake_version}" \
  --arg cmake_sha256 "${cmake_sha256}" \
  --arg cmake_path "${cmake_venv}/bin/cmake" \
  --arg boost_version "${boost_version}" \
  --arg boost_source_url "${boost_source_url}" \
  --arg boost_download_url "${boost_download_url}" \
  --arg boost_sha256 "${boost_sha256}" \
  --arg boost_prefix "${boost_prefix}" \
  --arg compiler "$(g++ --version | head -n 1)" \
  --arg finished_at "$(date --utc --iso-8601=seconds)" \
  '{version:1,status:"complete",phase:"7.1",git_sha:$git_sha,cmake:{version:$cmake_version,wheel_sha256:$cmake_sha256,path:$cmake_path},boost:{version:$boost_version,authoritative_source_url:$boost_source_url,download_mirror_url:$boost_download_url,source_sha256:$boost_sha256,prefix:$boost_prefix,libraries:["json","system"],linkage:"static"},compiler:$compiler,root:"/mnt/frontier-forge/tooling",finished_at:$finished_at}' \
  > results/phase7_1/a10_gateway_toolchain.json

echo "Phase 7.1 pinned CMake ${cmake_version} + Boost ${boost_version} toolchain is ready"
