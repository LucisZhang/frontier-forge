# Phase 3 human launch plan

Full runs are deliberately **not launched by agents**. Prepare and sync from the
local checkout, then bootstrap a single RTX 4090 Linux pod. First review and commit
the Phase 3 implementation. Clone or check out that exact commit on the pod; the
launcher refuses a dirty runtime code/config tree. `sync-up` transfers the prepared
data and receipts, not a substitute implementation snapshot.

```bash
make prepare-r1b
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

Planning estimates are 2h R0, 5h each R1 backend, 5h R2, 12h R3, and 15h per R4
seed: 74h core including the cross-check. Optional R1b adds 10h, for 84h planned.
These are planning guards, never reported as measured cost.
