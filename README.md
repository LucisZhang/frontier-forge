# frontier-forge

> Know your frontier. Then forge past it.

Post-train a 4B base LLM (SFT → distillation → DPO → GRPO) past the frontier-API
cost-quality frontier on a machine-verifiable structured-triage task, then serve it
through vLLM behind a custom C++20 LLM-aware gateway — with frozen eval sets,
confidence intervals, real cost accounting, and honest failure analysis.

Sequel to [nlp-eval-lab](https://github.com/LucisZhang/nlp-eval-lab), which mapped
the cost-quality frontier of classical / fine-tuned / frontier-API tiers. This
project pushes a small model across that frontier and productionizes the result.

**Status**: Phase 1.1 calibration remediation is implemented. Input contract v2,
scorer v2, label rules v2, unchanged split membership, and the fair-baseline prompt
are covered by local tests and audit artifacts. The receipt-backed API stand-in
scored 21.0% on 100 CAL rows, inside the locked 20–50% target band; see run
`phase1_1_api_calibration_v2_s20260815` in [results/runs.jsonl](results/runs.jsonl)
and the [calibration report](results/phase1_1_calibration_report.md). Human review is
required before Phase 2. Training and serving remain intentionally unimplemented.

## Local verification

```bash
uv sync --locked
make test
make gateway-test
SMOKE=1 make ingest
SMOKE=1 make splits
SMOKE=1 make calibrate-difficulty
make ci-lint
```

Execution details live in [PLAN.md](PLAN.md); locked decisions in
[DECISIONS.md](DECISIONS.md); agent rules in [AGENTS.md](AGENTS.md).
