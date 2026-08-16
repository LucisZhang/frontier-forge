# frontier-forge

> Know your frontier. Then forge past it.

Post-train a 4B base LLM (SFT → distillation → DPO → GRPO) past the frontier-API
cost-quality frontier on a machine-verifiable structured-triage task, then serve it
through vLLM behind a custom C++20 LLM-aware gateway — with frozen eval sets,
confidence intervals, real cost accounting, and honest failure analysis.

Sequel to [nlp-eval-lab](https://github.com/LucisZhang/nlp-eval-lab), which mapped
the cost-quality frontier of classical / fine-tuned / frontier-API tiers. This
project pushes a small model across that frontier and productionizes the result.

**Status**: Phase 2 is complete, and labels are frozen under D3.1. The final
stratified review recorded a 14% WRONG rate (four escalation and three refund
false negatives), while the unchanged 21.0% calibration re-score had no
discriminative power because none of its 100 labels changed. See the
[reviewer audit](results/phase1_2_label_audit.md), [freeze receipt](results/phase1_2_label_freeze.md),
and [re-score report](results/phase1_2_calibration_rescore.md).

The Phase 2 factory sent 6,000 deterministic frozen TRAIN records to
`anthropic/claude-haiku-4.5`, retained 1,450 matched SFT/DPO examples, and
quarantined 214 TEST-overlap hits. Exact account-reconciled API spend was
$12.700690 against the $20 cap; see the [data card](results/phase2_data_card.md)
and run `phase2_teacher_factory_v1_s20260816` in [results/runs.jsonl](results/runs.jsonl).
Phase 3's local implementation is now prepared: pinned R0–R4 configs, the optional
R1b 20k-rule-label ablation, TRL/Unsloth agreement gating, verifier-reward GRPO,
frozen paired evaluation, honest GPU ledgers, and separate BF16/GPTQ exports. Full
RTX 4090 runs and every resulting headline remain pending human launch; local smoke
records are excluded from the production `results/runs.jsonl` ledger. See the
[Phase 3 report](results/phase3_report.md) and [human launch plan](results/phase3_launch_plan.md).

## Local verification

```bash
uv sync --locked --group train
make test
make gateway-test
make phase1-2
SMOKE=1 make teacher-data
SMOKE=1 make teacher-audit
make teacher-audit
SMOKE=1 make phase3-smoke
make phase3-context-audit
SMOKE=1 make ingest
SMOKE=1 make splits
SMOKE=1 make calibrate-difficulty
make ci-lint
```

Execution details live in [PLAN.md](PLAN.md); locked decisions in
[DECISIONS.md](DECISIONS.md); agent rules in [AGENTS.md](AGENTS.md).
