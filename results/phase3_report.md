# Phase 3 post-training ladder report

Mode: **FULL GPU**.

## Ladder

| Rung | Seed | Backend | Status | Task success | 95% CI | Schema valid | Tool accuracy | GPU hours | USD |
|---|---:|---|---|---:|---|---:|---:|---:|---:|
| R0 | 0 | trl | active | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.983 | $0.295 |
| R1 | 0 | trl | active | 66.3% | [64.2%, 68.4%] | 100.0% | 94.0% | 3.479 | $1.044 |
| R1 | 0 | unsloth | active | 62.5% | [60.4%, 64.8%] | 100.0% | 92.5% | 1.281 | $0.384 |
| R1B | 0 | trl | active | 99.1% | [98.6%, 99.5%] | 100.0% | 99.2% | 15.236 | $4.571 |
| R2 | 0 | trl | active | 52.1% | [50.0%, 54.4%] | 100.0% | 84.3% | 3.606 | $1.082 |
| R3 | 0 | trl | active | 56.0% | [53.8%, 58.2%] | 99.9% | 87.2% | 1.925 | $0.578 |
| R4 | 0 | trl | superseded-inconclusive | 56.0% | [53.8%, 58.2%] | 99.9% | 87.2% | 2.108 | $0.632 |
| R4 | 0 | trl | active | 56.2% | [54.0%, 58.5%] | 99.9% | 87.3% | 1.859 | $0.558 |
| R4 | 1 | trl | superseded-inconclusive | 56.0% | [53.8%, 58.2%] | 99.9% | 87.2% | 1.529 | $0.459 |
| R4 | 1 | trl | active | 56.2% | [53.9%, 58.5%] | 99.9% | 87.3% | 1.661 | $0.498 |
| R4 | 2 | trl | superseded-inconclusive | 56.0% | [53.8%, 58.2%] | 99.9% | 87.2% | 1.526 | $0.458 |

## Adjacent paired deltas

- R0 → R1: 66.3% paired delta, 95% CI [64.4%, 68.3%], n=2000.
- R1 → R2: -14.2% paired delta, 95% CI [-15.7%, -12.6%], n=2000.
- R2 → R3: 3.8% paired delta, 95% CI [2.6%, 5.1%], n=2000.
- R3 → R4: 0.2% paired delta, 95% CI [-0.1%, 0.7%], n=2000.

## R4 v2 fresh-pool verdict

- Seed 0 vs R3: 0.2% paired delta, 95% CI [-0.1%, 0.7%], n=2000.
- Seed 1 vs R3: 0.2% paired delta, 95% CI [-0.1%, 0.7%], n=2000.
- Seed 2: aborted by the locked ten-step reward-variance guard; no frozen evaluation or paired delta exists.
- Final R4 v2 verdict: **ABORTED BY LOCKED GUARD**; seed(s) 2 stopped before frozen evaluation. The completed seed deltas are retained but not aggregated, and no retry or reward tuning is authorized.
- Seed 0 reward-variance gate: `passed-nonzero-reward-variance`; opening observations=10, passed=true.
- Seed 1 reward-variance gate: `passed-nonzero-reward-variance`; opening observations=10, passed=true.
- Seed 2 reward-variance gate: `aborted-zero-reward-variance`; opening observations=10, passed=false.

## Optional R1b ablation deltas

- R1 → R1B: 32.7% paired delta, 95% CI [30.6%, 34.5%], n=2000.
- R2 → R1B: 46.9% paired delta, 95% CI [44.8%, 48.9%], n=2000.

## Backend agreement

R1 TRL/Unsloth status: **agreement_failed**. Default after check: `trl`.

## Reward-hacking probes

- Length inflation reward increases: 0 rows.
- Markdown-format exploit reward increases: 0 rows.
- Model abstention rate: 0.1%; always-abstain task success: 0.4%.

## Failure and negative-result register

- GRPO incident: R4 seeds 0/1/2 from the original config are superseded-inconclusive. The missing `chat_template_kwargs={"enable_thinking": False}` left a trailing `</think>` prefix in rollouts; bare `json.loads` then returned reward 0.0 for every completion, producing zero reward variance and zero gradient. The fixed rerun disables thinking at chat-template rendering, strips through a trailing think block defensively, archives an opening rollout, and aborts after ten all-zero-variance steps.
- Phase 3.1 guard result: seed 0 proved the parser repair worked (clean bare JSON, no think marker, positive verifier reward), but the first ten optimizer steps all had mean reward 1.0, reward_std 0.0, frac_reward_zero_std 1.0, and grad_norm 0.0. In the archived opening window, all 24 completions across 6 prompt groups received reward 1.0 and advantage 0.0; observed variation was confined to secondary fields excluded from scorer-v2 reward. The guard aborted the run, and seeds 1/2 were not launched rather than bypassing the locked reward/data contract.
- Phase 3.2 guard result: seed 2 stopped after 10 opening steps all had zero within-group reward variance (mean verifier reward 0.940 across 80 completions). Per the locked plan, there is no retry, frozen-eval record, paired delta, or three-seed aggregate.
- R2 lost 14.2% task success versus R1.
- R0 trl seed 0 failure counts (nonexclusive): invalid_tool_arguments=2000, schema_invalid=2000, wrong_ambiguity=13, wrong_tool=2000, wrong_urgency=967.
- R1 trl seed 0 failure counts (nonexclusive): wrong_ambiguity=8, wrong_tool=119, wrong_urgency=634.
- R1 unsloth seed 0 failure counts (nonexclusive): wrong_ambiguity=8, wrong_tool=149, wrong_urgency=715.
- R1B trl seed 0 failure counts (nonexclusive): wrong_ambiguity=2, wrong_tool=17, wrong_urgency=11.
- R2 trl seed 0 failure counts (nonexclusive): schema_invalid=1, secondary_field_mismatch_only=79, wrong_ambiguity=8, wrong_tool=314, wrong_urgency=897.
- R3 trl seed 0 failure counts (nonexclusive): invalid_tool_arguments=1, schema_invalid=2, secondary_field_mismatch_only=74, wrong_ambiguity=8, wrong_tool=257, wrong_urgency=797.
- R4 trl seed 0 failure counts (nonexclusive): invalid_tool_arguments=1, schema_invalid=2, secondary_field_mismatch_only=76, wrong_ambiguity=8, wrong_tool=254, wrong_urgency=791.
- R4 trl seed 1 failure counts (nonexclusive): invalid_tool_arguments=1, schema_invalid=2, secondary_field_mismatch_only=76, wrong_ambiguity=8, wrong_tool=254, wrong_urgency=790.
- Failed GPU training attempt failed_cb153427dfb8b8ae6340: R0 seed 0, 0.048 GPU-hours, $0.014, exit code 0.
- Failed GPU training attempt failed_4bf236eb91be22c65963: R0 seed 0, 0.004 GPU-hours, $0.001, exit code 1.
- Failed GPU training attempt failed_473a7d89258a9b30f52c: R1 seed 0, 0.011 GPU-hours, $0.003, exit code 1.
- Failed GPU training attempt failed_0875f0f4e54bb56f80ec: R1 seed 0, 0.011 GPU-hours, $0.003, exit code 1.
- Failed GPU training attempt failed_490447c0d78bbbdf9ebe: R3 seed 0, 0.010 GPU-hours, $0.003, exit code 1.
- Failed GPU training attempt failed_089154b82e9781dc29f9: R4 seed 2, 0.035 GPU-hours, $0.010, exit code 130.
- Failed GPU training attempt failed_5a9dc15687f7738379c3: R4 seed 2, 0.018 GPU-hours, $0.005, exit code 130.
- Failed GPU training attempt failed_ccb9d60b8779c9232365: R4 seed 0, 0.354 GPU-hours, $0.106, exit code 1.
- Failed GPU training attempt failed_e2c802d85d0e409d82d3: R4 seed 2, 0.072 GPU-hours, $0.022, exit code 1.

## Export

- R1B trl seed 0: merged BF16 `7cf43a2905513f61797b78b7e3fd7ebdacd1cba4fc89abea9ce209401e6e6435`; GPTQ int4 `c99b42cf0e062cc75f2df8588725d0c29383666f3db0c1ae837ce15bfe6d39d2`.

## R4 best-seed export contract

R4 v2 stopped at the unchanged reward-variance guard; no export contract is opened and no tuning is authorized.

## Interpretation guards

R1 and R2 use the same 1,450 examples and identical decision-field labels after rejection sampling. Their difference is output phrasing, not label coverage.

The R2 loss therefore indicates that the teacher's semantic phrasing transferred a policy prior that diverged from the frozen keyword rules on new inputs; before filtering, teacher/rule urgency agreement was 36.8%.

Boundary condition: this project shows distillation adds no value when perfect rule-generated labels are free and unlimited. It does not generalize to settings where gold labels are scarce and no executable labeling rules exist; there, teacher quality is decisive. A more semantic Sonnet-class teacher would be expected to diverge further from this keyword policy, not close the measured gap.

## Draft headline

Draft: scaling free rule labels from 1,450 to 20,000 examples raised task success from 66.3% to 99.1% (+32.7%, paired 95% CI [30.6%, 34.5%]) for 15.236 measured RTX4090 GPU-hours ($4.571).

## Reproduction

```bash
python -m forge.train.report
```

All task-success intervals use 1,000 fixed-seed bootstrap resamples. Field F1 entries are single-label micro-F1 (equivalent to exact-match accuracy). Urgency ground truth is the frozen rule policy, not human semantic judgment.
