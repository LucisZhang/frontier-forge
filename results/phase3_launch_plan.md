# Phase 3 human launch plan

Full runs are launched by the human unless the current session explicitly delegates
the remote work. Prepare and sync from the local checkout, then use a single RTX 4090
Linux pod. First review and commit
the Phase 3 implementation. Clone or check out that exact commit on the pod; the
launcher refuses a dirty runtime code/config tree. `sync-up` transfers the prepared
data and receipts, not a substitute implementation snapshot.

```bash
make prepare-r1b
make prepare-r4-v2
make phase3-context-audit
make phase3-preflight
FORGE_REMOTE_ROOT=user@host:/path/to/frontier-forge make sync-up
# On the pod, from the synced checkout:
scripts/remote/bootstrap.sh
export FORGE_GPU_HOURLY_USD=0.42  # example only; replace with the actual pod rate
```

Bootstrap creates two environments from the same universal lock: `.venv` is the
canonical TRL 1.10 reference/evaluation/export stack, while `.venv-unsloth` is the
isolated TRL 0.24 stack required by the pinned Unsloth release. The launcher uses
the latter only for Unsloth training; every adapter is evaluated in `.venv`.

Run the commands below only on that pod.
Each command starts a resumable tmux task, records actual wall-clock GPU cost on
success or failure, and stops before a run whose projected total would exceed 90h.
`FORGE_GPU_HOURLY_USD` is mandatory and must be the rental rate shown for that pod;
the `$0.50/hour` values in rung configs are planning assumptions only.

Reference and backend gate:

```bash
scripts/remote/launch_phase3.sh r0 0 trl
scripts/remote/launch_phase3.sh r1 0 trl
scripts/remote/launch_phase3.sh r1 0 unsloth
```

The third command records the paired R1 agreement gate. Only if its 95% paired
bootstrap CI includes zero does `auto` select Unsloth for later rungs.

Core ladder after the agreement gate:

```bash
scripts/remote/launch_phase3.sh r2 0 auto
scripts/remote/launch_phase3.sh r3 0 auto
scripts/remote/launch_phase3.sh r4 0 auto
scripts/remote/launch_phase3.sh r4 1 auto
scripts/remote/launch_phase3.sh r4 2 auto
```

Optional R1b, only while the actual ledger leaves room in the 60–90h envelope:

```bash
scripts/remote/launch_phase3.sh r1b 0 auto
```

Phase 3.1 reruns R4 on the TRL reference path under the versioned
`phase3_1_reward_fix` artifact root, then exports the already-trained R1b seed-0
adapter. On a shared pod, inspect `nvidia-smi` before every command; both launchers
also fail closed when any compute process is present.

```bash
export FORGE_GPU_HOURLY_USD=0.30
scripts/remote/launch_phase3.sh r4 0 trl
scripts/remote/launch_phase3.sh r4 1 trl
scripts/remote/launch_phase3.sh r4 2 trl
scripts/remote/launch_phase3_export.sh 0 trl
```

Phase 3.2 supersedes the active R4 contract with the versioned
`phase3_2_fresh_pool` run root. The 8,000-row pool must be materialized and synced
before launch. Run seeds sequentially on TRL. Immediately before each launch, inspect
`nvidia-smi`; on a shared pod, wait if any process is using the GPU. Use only the
`forge-r4-trl-s*` tmux sessions and never stop or reboot the pod.

```bash
make prepare-r4-v2
make phase3-context-audit
export FORGE_GPU_HOURLY_USD=0.30
scripts/remote/launch_phase3.sh r4 0 trl
scripts/remote/launch_phase3.sh r4 1 trl
scripts/remote/launch_phase3.sh r4 2 trl
```

The unchanged ten-step variance guard is fail-closed. If any v2 seed aborts there,
stop and report instead of tuning. An R4 export contract opens only when a seed's
paired delta versus R3 has a 95% CI whose lower bound is greater than zero.

The Phase 3.2 config projects 3h per R4 v2 seed. Preflight combines that next-run
projection with the append-only measured ledger and refuses a launch above 90h.
Planning values are never reported as measured cost.
