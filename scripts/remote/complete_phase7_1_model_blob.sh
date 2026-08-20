#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 7.1 model blob recovery is Linux-only" >&2
  exit 2
fi
: "${FORGE_BENCH_GIT_SHA:?Set FORGE_BENCH_GIT_SHA to the benchmark commit}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(git rev-parse HEAD)" != "${FORGE_BENCH_GIT_SHA}" ]]; then
  echo "checked-out commit differs from FORGE_BENCH_GIT_SHA" >&2
  exit 2
fi
if tmux has-session -t phase7-bootstrap 2>/dev/null; then
  echo "stop the Phase 7.1 bootstrap session before completing its partial blob" >&2
  exit 2
fi

blob_sha256=b4d18ccadf1231d2e89a365449690d6494a620ed6d78ccafd8a1dfbb22d4c58d
blob_bytes=3991298872
revision=fd4ae1e1989dcb1641a496bf796031491518983e
model_file=bf16-mtp-preserved/model-00001-of-00003.safetensors
endpoint=https://hf-mirror.com
url="${endpoint}/Luciss007/frontier-forge-r1b/resolve/${revision}/${model_file}"
cache_root="${FORGE_CACHE_ROOT:-/mnt/frontier-forge/cache}"
blob_dir="${cache_root}/huggingface/hub/models--Luciss007--frontier-forge-r1b/blobs"
partial_blob="${blob_dir}/${blob_sha256}.incomplete"
final_blob="${blob_dir}/${blob_sha256}"
jobs="${FORGE_MODEL_RANGE_JOBS:-8}"
chunk_bytes="${FORGE_MODEL_RANGE_BYTES:-67108864}"
if [[ ! "${jobs}" =~ ^[1-9][0-9]*$ ]] || (( jobs > 16 )); then
  echo "FORGE_MODEL_RANGE_JOBS must be an integer from 1 through 16" >&2
  exit 2
fi
if [[ ! "${chunk_bytes}" =~ ^[1-9][0-9]*$ ]] || (( chunk_bytes < 1048576 )); then
  echo "FORGE_MODEL_RANGE_BYTES must be an integer of at least 1048576" >&2
  exit 2
fi
mkdir -p "${blob_dir}"
if [[ -f "${final_blob}" ]]; then
  test "$(stat -c %s "${final_blob}")" = "${blob_bytes}"
  printf '%s  %s\n' "${blob_sha256}" "${final_blob}" | sha256sum -c -
  echo "Phase 7.1 fixed model blob is already complete"
  exit 0
fi
if [[ ! -f "${partial_blob}" ]]; then
  echo "the resumable Hugging Face partial blob is missing: ${partial_blob}" >&2
  exit 2
fi
prefix_bytes="$(stat -c %s "${partial_blob}")"
if (( prefix_bytes <= 0 || prefix_bytes >= blob_bytes )); then
  echo "partial blob size is outside the resumable range: ${prefix_bytes}" >&2
  exit 2
fi

segment_root="${cache_root}/model-segments/${blob_sha256}/${prefix_bytes}"
segment_dir="${segment_root}/parts"
ranges="${segment_root}/ranges.tsv"
segment_manifest="${segment_root}/segments.sha256"
mkdir -p "${segment_dir}"
.venv-phase4/bin/python - "${prefix_bytes}" "${blob_bytes}" "${chunk_bytes}" "${ranges}" <<'PY'
from pathlib import Path
import sys

start = int(sys.argv[1])
total = int(sys.argv[2])
chunk = int(sys.argv[3])
path = Path(sys.argv[4])
rows = []
index = 0
while start < total:
    end = min(start + chunk, total) - 1
    rows.append(f"{index:04d}\t{start}\t{end}\t{end - start + 1}\n")
    index += 1
    start = end + 1
path.write_text("".join(rows))
print(f"planned {len(rows)} byte ranges")
PY

export url segment_dir
xargs -P "${jobs}" -n 4 bash -c '
  set -euo pipefail
  index="$1"
  start="$2"
  end="$3"
  expected_bytes="$4"
  part="${segment_dir}/${index}-${start}-${end}.part"
  if [[ -f "${part}" ]] && [[ "$(stat -c %s "${part}")" = "${expected_bytes}" ]]; then
    exit 0
  fi
  temporary="${part}.tmp"
  curl -L -sS --fail --retry 12 --retry-all-errors --retry-delay 2 \
    --connect-timeout 10 --max-time 600 --range "${start}-${end}" \
    -o "${temporary}" "${url}"
  if [[ "$(stat -c %s "${temporary}")" != "${expected_bytes}" ]]; then
    echo "range ${start}-${end} returned an unexpected byte count" >&2
    exit 1
  fi
  mv "${temporary}" "${part}"
' _ < "${ranges}"

while IFS=$'\t' read -r index start end expected_bytes; do
  part="${segment_dir}/${index}-${start}-${end}.part"
  test "$(stat -c %s "${part}")" = "${expected_bytes}"
done < "${ranges}"
find "${segment_dir}" -maxdepth 1 -type f -name '*.part' -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${segment_manifest}"
segment_manifest_sha256="$(sha256sum "${segment_manifest}" | cut -c1-64)"
prefix_sha256="$(sha256sum "${partial_blob}" | cut -c1-64)"

assembled="${blob_dir}/${blob_sha256}.assembled"
.venv-phase4/bin/python - "${partial_blob}" "${ranges}" "${segment_dir}" "${assembled}" <<'PY'
from pathlib import Path
import shutil
import sys

prefix = Path(sys.argv[1])
ranges = Path(sys.argv[2])
parts = Path(sys.argv[3])
assembled = Path(sys.argv[4])
with assembled.open("wb") as output:
    with prefix.open("rb") as source:
        shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    for row in ranges.read_text().splitlines():
        index, start, end, _expected = row.split("\t")
        part = parts / f"{index}-{start}-{end}.part"
        with part.open("rb") as source:
            shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
PY
test "$(stat -c %s "${assembled}")" = "${blob_bytes}"
printf '%s  %s\n' "${blob_sha256}" "${assembled}" | sha256sum -c -
mv "${assembled}" "${final_blob}"

mkdir -p results/phase7_1
jq -n \
  --arg git_sha "${FORGE_BENCH_GIT_SHA}" \
  --arg endpoint "${endpoint}" \
  --arg revision "${revision}" \
  --arg model_file "${model_file}" \
  --arg blob_sha256 "${blob_sha256}" \
  --arg final_blob "${final_blob}" \
  --arg prefix_sha256 "${prefix_sha256}" \
  --arg ranges "${ranges}" \
  --arg segment_manifest "${segment_manifest}" \
  --arg segment_manifest_sha256 "${segment_manifest_sha256}" \
  --argjson blob_bytes "${blob_bytes}" \
  --argjson prefix_bytes "${prefix_bytes}" \
  --argjson jobs "${jobs}" \
  --argjson chunk_bytes "${chunk_bytes}" \
  --arg finished_at "$(date --utc --iso-8601=seconds)" \
  '{version:1,status:"complete",phase:"7.1",git_sha:$git_sha,transport:{endpoint:$endpoint,method:"parallel_http_range_resume",jobs:$jobs,chunk_bytes:$chunk_bytes},identity:{fixed_revision:$revision,model_file:$model_file,lfs_sha256:$blob_sha256,bytes:$blob_bytes},retained_prefix:{bytes:$prefix_bytes,sha256:$prefix_sha256},segments:{ranges:$ranges,manifest:$segment_manifest,manifest_sha256:$segment_manifest_sha256},final_blob:$final_blob,finished_at:$finished_at}' \
  > results/phase7_1/a10_model_blob_acceleration.json
echo "Phase 7.1 fixed model blob completed and verified with parallel HTTP ranges"
