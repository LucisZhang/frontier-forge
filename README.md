# frontier-forge

> Know your frontier. Then forge past it.

Post-train a 4B base LLM (SFT → distillation → DPO → GRPO) past the frontier-API
cost-quality frontier on a machine-verifiable structured-triage task, then serve it
through vLLM behind a custom C++20 LLM-aware gateway — with frozen eval sets,
confidence intervals, real cost accounting, and honest failure analysis.

Sequel to [nlp-eval-lab](https://github.com/LucisZhang/nlp-eval-lab), which mapped
the cost-quality frontier of classical / fine-tuned / frontier-API tiers. This
project pushes a small model across that frontier and productionizes the result.

**Status**: Phase 1.2 is implemented. Label rules v3 scope escalation/refund
keywords to complaint narratives, assert configured tool precedence, and preserve
all four frozen split memberships. Re-derivation changed 5,762 labels, including
5,740 strong-action transitions; see the [reviewer audit](results/phase1_2_label_audit.md).
The existing 100 API receipts remain at 21.0% after the offline v3 re-score (0 new
calls); see run `phase1_2_api_calibration_v3_offline_rescore_s20260815` in
[results/runs.jsonl](results/runs.jsonl) and the
[re-score report](results/phase1_2_calibration_rescore.md). Human review is required
before label freeze or Phase 2. Training and serving remain intentionally
unimplemented.

## Local verification

```bash
uv sync --locked
make test
make gateway-test
make phase1-2
SMOKE=1 make ingest
SMOKE=1 make splits
SMOKE=1 make calibrate-difficulty
make ci-lint
```

Execution details live in [PLAN.md](PLAN.md); locked decisions in
[DECISIONS.md](DECISIONS.md); agent rules in [AGENTS.md](AGENTS.md).
