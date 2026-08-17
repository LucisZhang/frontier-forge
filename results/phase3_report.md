# Phase 3 post-training ladder report

Mode: **FULL GPU**.

## Ladder

| Rung | Seed | Backend | Task success | 95% CI | Schema valid | Tool accuracy | GPU hours | USD |
|---|---:|---|---:|---|---:|---:|---:|---:|
| R0 | 0 | trl | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.983 | $0.295 |
| R1 | 0 | trl | 66.3% | [64.2%, 68.4%] | 100.0% | 94.0% | 3.479 | $1.044 |
| R1 | 0 | unsloth | 62.5% | [60.4%, 64.8%] | 100.0% | 92.5% | 1.281 | $0.384 |
| R1B | — | — | optional; pending | — | — | — | — | — |
| R2 | 0 | trl | 52.1% | [50.0%, 54.4%] | 100.0% | 84.3% | 3.606 | $1.082 |
| R3 | 0 | trl | 56.0% | [53.8%, 58.2%] | 99.9% | 87.2% | 1.925 | $0.578 |
| R4 | 0 | trl | 56.0% | [53.8%, 58.2%] | 99.9% | 87.2% | 2.108 | $0.632 |
| R4 | 1 | trl | 56.0% | [53.8%, 58.2%] | 99.9% | 87.2% | 1.529 | $0.459 |

## Adjacent paired deltas

- R0 → R1: 66.3% paired delta, 95% CI [64.4%, 68.3%], n=2000.
- R1 → R2: -14.2% paired delta, 95% CI [-15.7%, -12.6%], n=2000.
- R2 → R3: 3.8% paired delta, 95% CI [2.6%, 5.1%], n=2000.
- R3 → R4: 0.0% paired delta, 95% CI [0.0%, 0.0%], n=2000.

## Optional R1b ablation deltas

- R1 → R1B: pending.
- R2 → R1B: pending.

## Backend agreement

R1 TRL/Unsloth status: **agreement_failed**. Default after check: `trl`.

## Reward-hacking probes

- Length inflation reward increases: 0 rows.
- Markdown-format exploit reward increases: 0 rows.
- Model abstention rate: 0.1%; always-abstain task success: 0.4%.

## Failure and negative-result register

- R2 lost 14.2% task success versus R1.
- R0 seed 0 failure counts (nonexclusive): invalid_tool_arguments=2000, schema_invalid=2000, wrong_ambiguity=13, wrong_tool=2000, wrong_urgency=967.
- R1 seed 0 failure counts (nonexclusive): wrong_ambiguity=8, wrong_tool=119, wrong_urgency=634.
- R1 seed 0 failure counts (nonexclusive): wrong_ambiguity=8, wrong_tool=149, wrong_urgency=715.
- R2 seed 0 failure counts (nonexclusive): schema_invalid=1, secondary_field_mismatch_only=79, wrong_ambiguity=8, wrong_tool=314, wrong_urgency=897.
- R3 seed 0 failure counts (nonexclusive): invalid_tool_arguments=1, schema_invalid=2, secondary_field_mismatch_only=74, wrong_ambiguity=8, wrong_tool=257, wrong_urgency=797.
- R4 seed 0 failure counts (nonexclusive): invalid_tool_arguments=1, schema_invalid=2, secondary_field_mismatch_only=74, wrong_ambiguity=8, wrong_tool=257, wrong_urgency=797.
- R4 seed 1 failure counts (nonexclusive): invalid_tool_arguments=1, schema_invalid=2, secondary_field_mismatch_only=74, wrong_ambiguity=8, wrong_tool=257, wrong_urgency=797.
- Failed GPU attempt failed_cb153427dfb8b8ae6340: R0 seed 0, 0.048 GPU-hours, $0.014, exit code 0.
- Failed GPU attempt failed_4bf236eb91be22c65963: R0 seed 0, 0.004 GPU-hours, $0.001, exit code 1.
- Failed GPU attempt failed_473a7d89258a9b30f52c: R1 seed 0, 0.011 GPU-hours, $0.003, exit code 1.
- Failed GPU attempt failed_0875f0f4e54bb56f80ec: R1 seed 0, 0.011 GPU-hours, $0.003, exit code 1.
- Failed GPU attempt failed_490447c0d78bbbdf9ebe: R3 seed 0, 0.010 GPU-hours, $0.003, exit code 1.

## Export

- trl seed 0: merged BF16 `1304ebf174da98f3b761f482bd2d88b58d66e30bc9cef66abb6104ab70ef6bbc`; GPTQ int4 `3d3bed316134e3d583b7be16a8a89d4de2bf2f40c34365623d0e1c2edec10958`.

## Draft headline

Withheld until the full ladder, three R4 seeds, CIs, and costs are recorded.

## Reproduction

```bash
python -m forge.train.report
```

All task-success intervals use 1,000 fixed-seed bootstrap resamples. Field F1 entries are single-label micro-F1 (equivalent to exact-match accuracy). Urgency ground truth is the frozen rule policy, not human semantic judgment.
