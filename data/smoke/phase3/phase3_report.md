# Phase 3 post-training ladder report

Mode: **SMOKE (non-headline)**.

## Ladder

| Rung | Seed | Backend | Status | Task success | 95% CI | Schema valid | Tool accuracy | GPU hours | USD |
|---|---:|---|---|---:|---|---:|---:|---:|---:|
| R0 | 0 | trl | active | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.000 | $0.000 |
| R1 | 0 | trl | active | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.000 | $0.000 |
| R1B | 0 | trl | active | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.000 | $0.000 |
| R2 | 0 | trl | active | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.000 | $0.000 |
| R3 | 0 | trl | active | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.000 | $0.000 |
| R4 | 0 | trl | active | 0.0% | [0.0%, 0.0%] | 0.0% | 0.0% | 0.000 | $0.000 |

## Adjacent paired deltas

- R0 → R1: 0.0% paired delta, 95% CI [0.0%, 0.0%], n=4.
- R1 → R2: 0.0% paired delta, 95% CI [0.0%, 0.0%], n=4.
- R2 → R3: 0.0% paired delta, 95% CI [0.0%, 0.0%], n=4.
- R3 → R4: 0.0% paired delta, 95% CI [0.0%, 0.0%], n=4.

## Optional R1b ablation deltas

- R1 → R1B: 0.0% paired delta, 95% CI [0.0%, 0.0%], n=4.
- R2 → R1B: 0.0% paired delta, 95% CI [0.0%, 0.0%], n=4.

## Backend agreement

R1 TRL/Unsloth status: **agreement_failed**. Default after check: `trl`.

## Reward-hacking probes

- Length inflation reward increases: 0 rows.
- Markdown-format exploit reward increases: 0 rows.
- Model abstention rate: 0.0%; always-abstain task success: 0.0%.

## Failure and negative-result register

- GRPO incident: R4 seeds 0/1/2 from the original config are superseded-inconclusive. The missing `chat_template_kwargs={"enable_thinking": False}` left a trailing `</think>` prefix in rollouts; bare `json.loads` then returned reward 0.0 for every completion, producing zero reward variance and zero gradient. The fixed rerun disables thinking at chat-template rendering, strips through a trailing think block defensively, archives an opening rollout, and aborts after ten all-zero-variance steps.
- R0 trl seed 0 failure counts (nonexclusive): invalid_json=4, invalid_tool_arguments=4, wrong_ambiguity=4, wrong_tool=4, wrong_urgency=4.
- R1 trl seed 0 failure counts (nonexclusive): invalid_json=4, invalid_tool_arguments=4, wrong_ambiguity=4, wrong_tool=4, wrong_urgency=4.
- R1B trl seed 0 failure counts (nonexclusive): invalid_json=4, invalid_tool_arguments=4, wrong_ambiguity=4, wrong_tool=4, wrong_urgency=4.
- R2 trl seed 0 failure counts (nonexclusive): invalid_json=4, invalid_tool_arguments=4, wrong_ambiguity=4, wrong_tool=4, wrong_urgency=4.
- R3 trl seed 0 failure counts (nonexclusive): invalid_json=4, invalid_tool_arguments=4, wrong_ambiguity=4, wrong_tool=4, wrong_urgency=4.
- R4 trl seed 0 failure counts (nonexclusive): invalid_json=4, invalid_tool_arguments=4, wrong_ambiguity=4, wrong_tool=4, wrong_urgency=4.

## Export

- SMOKE ONLY, R1B trl seed 0: adapter input `b52f331c73a152155afcee0fe73a7b2977fd40fc6831c73baa9d01493aa99691`; synthetic int4 packing `5d296c4cc41cfbbeea5812f256ada9ea668913a9ff71945a12acc91934105f66`. Neither is a deployable model export.

## R4 best-seed export contract

Pending all three active TRL reruns for seeds 0/1/2.

## Interpretation guards

R1 and R2 use the same 1,450 examples and identical decision-field labels after rejection sampling. Their difference is output phrasing, not label coverage.

The R2 loss therefore indicates that the teacher's semantic phrasing transferred a policy prior that diverged from the frozen keyword rules on new inputs; before filtering, teacher/rule urgency agreement was 36.8%.

Boundary condition: this project shows distillation adds no value when perfect rule-generated labels are free and unlimited. It does not generalize to settings where gold labels are scarce and no executable labeling rules exist; there, teacher quality is decisive. A more semantic Sonnet-class teacher would be expected to diverge further from this keyword policy, not close the measured gap.

## Draft headline

Draft: scaling free rule labels from 1,450 to 20,000 examples raised task success from 0.0% to 0.0% (+0.0%, paired 95% CI [0.0%, 0.0%]) for 0.000 measured RTX4090 GPU-hours ($0.000).

## Reproduction

```bash
python -m forge.train.report
```

All task-success intervals use 1,000 fixed-seed bootstrap resamples. Field F1 entries are single-label micro-F1 (equivalent to exact-match accuracy). Urgency ground truth is the frozen rule policy, not human semantic judgment.
