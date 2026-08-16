# Phase 3 post-training ladder report

Mode: **FULL GPU**.

Status: **REMOTE RUNS PENDING HUMAN LAUNCH**.

The implementation and launch contracts are prepared, but no full Phase 3 GPU metric, cost, backend-agreement result, or exported-weight hash exists yet. No headline is drafted from smoke data.

## Ladder

| Rung | Seed | Backend | Task success | 95% CI | Schema valid | Tool accuracy | GPU hours | USD |
|---|---:|---|---:|---|---:|---:|---:|---:|
| R0 | — | — | pending | — | — | — | — | — |
| R1 | — | — | pending | — | — | — | — | — |
| R1B | — | — | optional; pending | — | — | — | — | — |
| R2 | — | — | pending | — | — | — | — | — |
| R3 | — | — | pending | — | — | — | — | — |
| R4 | — | — | pending | — | — | — | — | — |

## Adjacent paired deltas

- R0 → R1: pending.
- R1 → R2: pending.
- R2 → R3: pending.
- R3 → R4: pending.

## Optional R1b ablation deltas

- R1 → R1B: pending.
- R2 → R1B: pending.

## Backend agreement

R1 TRL/Unsloth status: **pending_human_gpu_runs**. Default after check: `trl (reference)`.

## Reward-hacking probes

R4 probes are pending the human-launched GRPO run and frozen evaluation.

## Failure and negative-result register

No full-run negative result is available yet; none is inferred from smoke runs.

## Export

Merged BF16 and deployment GPTQ-int4 hashes are pending a human-launched R4 export.

## Draft headline

Withheld until the full ladder, three R4 seeds, CIs, and costs are recorded.

## Reproduction

```bash
python -m forge.train.report
```

All task-success intervals use 1,000 fixed-seed bootstrap resamples. Field F1 entries are single-label micro-F1 (equivalent to exact-match accuracy). Urgency ground truth is the frozen rule policy, not human semantic judgment.
