# frontier-forge

> Know your frontier. Then forge past it.

Post-train a 4B base LLM (SFT → distillation → DPO → GRPO) past the frontier-API
cost-quality frontier on a machine-verifiable structured-triage task, then serve it
through vLLM behind a custom C++20 LLM-aware gateway — with frozen eval sets,
confidence intervals, real cost accounting, and honest failure analysis.

Sequel to [nlp-eval-lab](https://github.com/LucisZhang/nlp-eval-lab), which mapped
the cost-quality frontier of classical / fine-tuned / frontier-API tiers. This
project pushes a small model across that frontier and productionizes the result.

**Status**: Phase 1 task machinery is implemented locally: the C1 schema,
deterministic rule labels and verifier, pinned-snapshot ingest, frozen temporal
splits, human-audit artifacts, and calibration plumbing. Human review is still
required before accepting the label candidate or final difficulty settings. Training
and serving remain intentionally unimplemented.

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
