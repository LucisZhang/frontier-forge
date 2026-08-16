# Phase 3 post-training ladder report

Mode: **FULL GPU**.

## Ladder

| Rung | Seed | Backend | Task success | 95% CI | Schema valid | Tool accuracy | GPU hours | USD |
|---|---:|---|---:|---|---:|---:|---:|---:|
| R0 | 0 | trl | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.983 | $0.295 |
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

- R0 seed 0 failure counts (nonexclusive): invalid_tool_arguments=2000, schema_invalid=2000, wrong_ambiguity=13, wrong_tool=2000, wrong_urgency=967.
- Failed GPU attempt failed_cb153427dfb8b8ae6340: R0 seed 0, 0.048 GPU-hours, $0.014, exit code 0.
- Failed GPU attempt failed_4bf236eb91be22c65963: R0 seed 0, 0.004 GPU-hours, $0.001, exit code 1.

## Export

Merged BF16 and deployment GPTQ-int4 hashes are pending a human-launched R4 export.

## Draft headline

Withheld until the full ladder, three R4 seeds, CIs, and costs are recorded.

## Reproduction

```bash
python -m forge.train.report
```

All task-success intervals use 1,000 fixed-seed bootstrap resamples. Field F1 entries are single-label micro-F1 (equivalent to exact-match accuracy). Urgency ground truth is the frozen rule policy, not human semantic judgment.
