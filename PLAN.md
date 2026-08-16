# PLAN.md — frontier-forge phased execution spec

Audience: coding agents (Codex). Read `AGENTS.md` first; locked decisions are in
`DECISIONS.md` (D1–D10) and are referenced below as e.g. [D3].

## How this plan is executed

- One phase per agent thread. Kickoff prompt pattern:
  `Read AGENTS.md, DECISIONS.md, and PLAN.md §Phase N. Execute Phase N only. Report the gate checklist when done.`
- A phase is DONE only when every gate item is checked with evidence (command
  output or file path). Failed items are reported as failed, not reworded.
- Phases 0–2 are local-only. Phases 3–4 produce remote-ready scripts locally
  (SMOKE=1 green first); full runs are launched by the human [AGENTS.md Hardware Rules].
- Phase 5 may run in parallel with Phases 3–4 (separate thread, no shared files).

## Shared contracts (all phases)

### C1. Task output schema (v1, [D3])
Every model/verifier component uses exactly this JSON schema (defined once in
`src/forge/verify/schema.py`, exported to `configs/task_schema.json`):

```json
{
  "product": "<enum: mirrors CFPB product taxonomy, defined in Phase 1>",
  "issue": "<string, normalized>",
  "company": "<string | null>",
  "urgency": "low | medium | high",
  "ambiguity_flag": true,
  "tool_call": {
    "name": "escalate_to_regulator | request_more_info | route_to_company | start_refund_workflow | close_no_action",
    "arguments": { "<per-tool schema, defined in Phase 1>": "..." }
  }
}
```
`ambiguity_flag=true` + `tool_call.name=request_more_info` is the abstention path;
the verifier rewards correct abstention and penalizes confident wrong answers.

### C2. runs.jsonl record schema (append-only, [D10])
```json
{
  "run_id": "r1_sft_rule_s0",
  "phase": 3,
  "config_path": "configs/r1_sft_rule.yaml",
  "config_hash": "…", "git_sha": "…", "dataset_hash": "…",
  "model": "…", "seed": 0,
  "metrics": { "task_success": 0.0, "schema_valid": 0.0, "tool_acc": 0.0,
               "field_f1": {}, "abstain_correct": 0.0, "ci95": [0.0, 0.0] },
  "cost": { "gpu_type": "RTX4090", "gpu_hours": 0.0, "usd": 0.0, "api_usd": 0.0 },
  "started_at": "<passed in by launcher, never generated in-code from wall clock alone>",
  "finished_at": "…",
  "notes": ""
}
```

### C3. Makefile target inventory
Every target accepts `SMOKE=1`. Targets marked ® are remote-only in full mode.

```
test lint gateway-test gateway-tsan
ingest splits calibrate-difficulty
teacher-data teacher-audit
train-sft train-dpo train-grpo ®      eval ®(full) export-model ®
serve-bench ® spec-decode-bench ® structured-bench ® bench-report
gateway-bench ®
sync-up sync-down    # rsync whitelists: data manifests, configs, results, checkpoints
demo-build reproduce-headline
```

---

## Phase 0 — Scaffold (local, 1–2 days)

**Goal**: empty-but-green repository skeleton; no business logic.

**Deliverables**
- `uv` project (Python 3.12, locked), directory tree exactly as in AGENTS.md
  §Project Structure, with `__init__.py` stubs.
- Makefile with every C3 target present; unimplemented ones exit 0 with
  `"[stub] <target>"`; `SMOKE=1` plumbing works end-to-end on a no-op pipeline.
- `gateway/` CMake hello-world: builds with clang locally, GoogleTest wired,
  ASan/UBSan flags in a `sanitize` preset; `gateway-tsan` runs in a Linux Docker image.
- CI (GitHub Actions): pytest + ruff + gateway build on a 50-row fixture. No GPU steps.
- `scripts/remote/bootstrap.sh` (pod setup: uv, CUDA wheel install, tmux) and
  `scripts/remote/sync.sh` (rsync whitelists) — written, not yet exercised.
- `.gitignore` covering `.env`, `data/`, `checkpoints/`, `*.gguf`, wheels.

**Verification**: `make test`, `make gateway-test`, `SMOKE=1 make ingest` (no-op),
CI config passes `act` dry-run or YAML lint.

**Gate**
- [ ] `make test` green  - [ ] `make gateway-test` green
- [ ] every C3 target exists  - [ ] CI green on first push
- [ ] no phase-1+ logic implemented

**Constraints**: do not design the verifier, do not download datasets.

---

## Phase 1 — Task spec, verifier, frozen splits (local + small API budget, 3–5 days)

**Goal**: the ground truth machinery everything else stands on.

**Deliverables**
1. `src/forge/verify/schema.py`: C1 schema finalized — product enum from CFPB
   taxonomy, the 5 tool argument schemas, exported `configs/task_schema.json`.
2. `src/forge/data/ingest.py`: reuse nlp-eval-lab's frozen CFPB snapshot
   (path/manifest read from `configs/data_sources.yaml`; never re-download if the
   snapshot hash matches). Derive labels: rule-based mapping for `product`/`issue`/
   `company`; `urgency`/`ambiguity`/`tool_call` from a documented rule table +
   teacher spot-labels for ambiguous cases (≤$10 API budget, logged).
3. `src/forge/verify/verifier.py`: pure function
   `score(sample, model_output_json) -> ScoreBreakdown` covering: schema validity,
   per-field exact/normalized match, tool choice, tool arguments, abstention
   correctness. Deterministic, no network, no model calls.
4. Adversarial test suite `tests/test_verifier.py`: ≥40 cases — truncated JSON,
   extra fields, hallucinated enum values, wrong tool with plausible args,
   over-abstention, unicode/whitespace traps.
5. Splits: TRAIN / CAL / TEST-IID / TEST-DRIFT (temporal, mirror nlp-eval-lab
   convention). Materialize once → write manifest hashes → FROZEN [D10].
6. `make calibrate-difficulty`: run base-model zero-shot (SMOKE locally with a
   small API model stand-in; full run is the first remote job of Phase 3) and
   report the success band vs the 20–50% target [D3]. Output: calibration report
   markdown with the difficulty-knob settings chosen.

**Verification**: `make test` (verifier suite), `make splits` idempotent (second
run detects frozen manifests and no-ops), 200-sample human-audit file emitted to
`results/phase1_label_audit.md` for the human to review.

**Gate**
- [ ] ≥40 adversarial verifier tests green  - [ ] splits frozen with hashes
- [ ] label audit file delivered to human    - [ ] difficulty knobs documented
- [ ] zero network calls inside verifier

**Constraints**: difficulty-knob final values and any task-fallback decision [D3]
belong to the human; the agent only reports calibration evidence.

---

## Phase 1.1 — Calibration remediation (local + small API budget, 2–3 days)

**Why this phase exists**: Phase 1 calibration scored 0.0% because two of eight
task_success checks were unwinnable from the model's input (gold `issue`/`company`
come from metadata columns absent from the redacted narrative) and tool arguments
required verbatim template equality. See D3.1 for the human-approved fix.

**Goal**: implement input contract v2 [D3.1] so the calibration lands in the
20–50% band for a real reason.

**Deliverables**
1. Verifier v2: task_success = urgency + ambiguity_flag + tool_choice +
   structural tool-argument validity (schema/keys/types; no verbatim free-text
   matching). `issue`/`company` normalized match moves to a secondary metrics
   block. Version the scorer (`scorer_version: 2`) in every ScoreBreakdown.
2. Input builder: model input = narrative + source product/issue/company fields;
   used consistently by calibration, and later by all training/eval [D3.1].
3. Label rules v2 (`configs/label_rules.yaml` version bump): fix the ambiguity
   rule so phrase triggers (e.g. "not sure") cannot alone flag long narratives
   (documented design; re-run the 200-row audit emit for the changed rows).
4. Fair-baseline calibration prompt v2: full task spec (urgency policy, ambiguity
   definition, tool registry semantics) included; re-run calibration on 100–200
   CAL rows with the API stand-in (budget ≤$15, receipted).
5. Updated adversarial verifier tests covering the v2 scoring (target ≥60 cases
   total, including: correct decision + wrong issue text → task_success true;
   verbatim-template argument no longer required; long-narrative "not sure" no
   longer auto-ambiguous).
6. `results/phase1_1_calibration_report.md`: new success band, CI, per-check
   breakdown, and a delta explanation vs the Phase 1 report.
7. Git hygiene: commit Phase 1 as delivered (baseline), then Phase 1.1 changes
   as separate commits; push and confirm CI green.

**Gate**
- [ ] calibration success in 20–50% band on n≥100 (or a documented escalation if
      still outside — do NOT tune knobs silently to force the band)
- [ ] verifier v2 tests green (≥60 cases)  - [ ] scorer_version stamped
- [ ] label rules v2 documented + audit rows re-emitted for human review
- [ ] split membership hashes unchanged  - [ ] CI green after push

**Constraints**: split membership is untouchable; label re-derivation allowed only
per D3.1. No Phase 2 work. The human reviews the re-emitted audit rows and the new
band before Phase 2 starts.

---

## Phase 1.2 — Escalation-rule fix (local, ~half day)

**Why**: human-delegated review of the strong-action rows found the escalation
trigger matches keywords over `issue + narrative` concatenation, so CFPB *product
taxonomy strings* containing "identity theft" (e.g. "Credit monitoring or identity
theft protection services") mislabel routine service complaints as
`escalate_to_regulator` (~5,737 rows population-wide; 82/813 of the v2-changed
escalations), and their `reason: issue` argument emits the product name as a
nonsensical regulator reason. Must land before Phase 2 freezes labels.

**Deliverables** (label rules v3 + dataset hash bump; split membership untouched)
1. Scope escalation (and refund) keyword matching to the **narrative only**; stop
   matching against `source_issue`. Drop the two product-name issue strings and the
   legacy `Loan modification,collection,foreclosure` label from any trigger role.
2. `tools.priority` in configs/label_rules.yaml: either parse it and assert it
   matches the code's precedence, or delete it. No dead config.
3. Re-derive labels (v3), re-emit the changed-row audit; ADD a stratified
   strong-action sample (all changed escalate/refund rows up to 50, not
   hash-ranked luck) to the reviewer artifact.
4. Re-score the existing 100 calibration receipts offline against v3 labels (no
   new API calls) and report the band delta in
   `results/phase1_2_calibration_rescore.md`.
5. Document as known limitations (do NOT fix now): negation-blind matching
   (~18 changed rows contain negated "identity theft"), and the single-action
   taxonomy (escalation outranks refund, so dual-remedy complaints get one action).

**Gate**
- [ ] v3 trigger scope narrative-only, product-name strings removed
- [ ] dead config resolved  - [ ] stratified strong-action audit emitted
- [ ] calibration re-score reported from existing receipts (still in/near band)
- [ ] split membership hashes unchanged  - [ ] tests green, CI green after push

**Constraints**: no new API spend; no Phase 2 work; label freeze happens only
after this phase's human review.

---

## Phase 2 — Teacher distillation data factory (local + API, 3–4 days)

**Goal**: two SFT corpora + DPO preference pairs, with a documented filter funnel.

**Deliverables**
1. `src/forge/teacher/generate.py`: teacher [D2] generates structured outputs for
   TRAIN inputs; every record carries teacher model id + prompt hash + raw response.
2. Filter funnel (each stage logged with retention %): verifier-scored rejection
   sampling → minhash dedup → contamination audit (n-gram overlap vs all TEST
   splits; any hit quarantines the sample).
3. Corpora: `sft_rule.jsonl` (Phase-1 rule labels) and `sft_distilled.jsonl`
   (teacher, filtered) — same input coverage so R1 vs R2 isolates data quality [D5].
4. DPO pairs: chosen = high-scoring teacher output; rejected = low-scoring teacher
   attempt or perturbed near-miss (perturbation taxonomy documented).
5. Data card `results/phase2_data_card.md`: sizes, funnel retention, contamination
   results, cost ledger (api_usd vs the $20–50 envelope [D6]), and a teacher-vs-rule
   disagreement breakdown per decision field (Phase 1.1 found urgency keyword-vs-
   semantics divergence: fair-prompted Haiku matched only 26% — quantify and discuss;
   downstream reports must state that urgency ground truth is rule policy, not human
   judgment).

**Verification**: `make teacher-data SMOKE=1` (10 samples, mock teacher) green;
`make teacher-audit` reproduces the funnel numbers from raw logs.

**Gate**
- [ ] both corpora + DPO pairs materialized with hashes
- [ ] contamination audit clean (or quarantine list committed)
- [ ] data card complete with cost ledger  - [ ] budget within envelope

**Constraints**: TEST splits are never sent to the teacher. Prompt templates are
versioned files in `configs/teacher_prompts/`, not inline strings.

---

## Phase 3 — Post-training ladder R0–R4 (remote GPU; smoke local; 1.5–2 weeks)

**Goal**: the [D5] ladder, every rung a runs.jsonl record with CIs and cost.

**Deliverables**
1. One YAML per rung in `configs/` (r0…r4), seeds pinned; `train-sft/dpo/grpo`
   launchers work in SMOKE=1 (0.5B model, ≤100 rows, Mac MPS/CPU) and full mode
   (pod, tmux, resumable, metrics JSON on exit).
2. TRL reference path first; one Unsloth cross-check run vs TRL on R1 must agree
   within CI before Unsloth becomes the default [D4].
3. GRPO (R4): reward = Phase-1 verifier score; log reward-hacking probes —
   length inflation, format exploitation, degenerate abstention rate.
4. Evaluation harness `make eval`: frozen TEST-IID + TEST-DRIFT, bootstrap CIs
   (n=1000, fixed seed), paired deltas between adjacent rungs, plus a small
   general-instruction regression probe (non-target capability drift).
5. `make export-model`: merge LoRA → full-precision weights + AWQ or GPTQ int4,
   both hashed and recorded [D10 quantization separation].
6. `results/phase3_report.md`: ladder table (metrics, CIs, GPU-hours, USD),
   failure-category analysis per rung, draft headline sentence(s).

**Verification**: all five rungs present in runs.jsonl (3 seeds for the headline
rung, 1 seed acceptable for non-headline rungs if budget-bound — record the
choice); `SMOKE=1 make train-sft && SMOKE=1 make eval` green on Mac.

**Gate**
- [ ] R0–R4 all recorded with cost fields  - [ ] adjacent-rung paired deltas + CIs
- [ ] Unsloth/TRL agreement check recorded - [ ] reward-hacking probes reported
- [ ] exported weights hashed (fp + int4)  - [ ] negative results (if any) in report

**Constraints**: the human launches full runs [AGENTS.md]. GPU-hour ledger updated
per run; if projected total exceeds the 60–90h envelope [D6], stop and report
before launching the next run.

---

## Phase 4 — Serving & inference engineering (remote GPU, ~1 week)

**Goal**: production-vocabulary numbers [D8] + the structured-output deep dive [D9].

**Deliverables**
1. `src/forge/bench/loadgen.py`: workload-controlled generator — fixed input/output
   length distributions, arrival-rate sweep, warm-up phase, client-side and
   server-side timing clearly separated.
2. `make serve-bench`: vLLM, best 2 variants × {BF16, int4}: TTFT, ITL, p50/p95
   e2e, tokens/s, req/s, VRAM, max stable concurrency, cost-per-1k-successful-tasks
   (successful = verifier-passing).
3. `make spec-decode-bench`: 0.5B draft; QPS sweep → win/lose boundary + acceptance
   rate [D8].
4. `make structured-bench` [D9]: (a) xGrammar vs Outlines overhead across schema
   complexity incl. cold-start compile; (b) constraint-tax reproduction — tool-call
   rate with schema constraint on/off; (c) two-pass decoupling mitigation with
   before/after task-success and latency.
5. Optional (only if W5 has slack): SGLang RadixAttention prefix-cache comparison,
   shared-system-prompt × 50-concurrency TTFT.
6. `make bench-report`: regenerates every figure/table in
   `results/phase4_serving_report.md` from raw run artifacts.

**Gate**
- [ ] disclosure block present (hardware/precision/load/arrival/warm-up/side)
- [ ] cost-per-1k-successful-tasks computed against verifier, not token counts
- [ ] spec-decode win/lose boundary plotted  - [ ] constraint-tax numbers + mitigation delta
- [ ] `make bench-report` idempotent

**Constraints**: no benchmark numbers enter any doc except via bench-report output.

---## Phase 5 — C++20 LLM-aware gateway (local dev; remote final bench; 2–3 weeks, parallel-safe)

**Goal**: [D7] admission-control gateway with measured overload behavior.

**Deliverables**
1. `gateway/` (standalone CMake): C++20 coroutines (Boost.Asio), OpenAI-compatible
   passthrough, SSE streaming proxy, deadline + cancellation propagation,
   connection pool, bounded request queue, token-aware admission control
   (estimated prompt/output tokens + live queue state → enqueue / degrade-route
   to fallback model / fast-reject with Retry-After), per-client rate limits,
   load shedding, circuit breaker, replica health checks, Prometheus metrics
   (queue depth, active requests, routing decisions, latency histograms).
2. Local dev upstreams: llama.cpp server (int4 4B) + `gateway/tests/mock_upstream`
   with injectable latency/errors/disconnects. All functional + failure-injection
   tests run on Mac.
3. Test matrix: GoogleTest unit + integration; ASan/UBSan locally; TSan in Docker.
4. Remote final bench `make gateway-bench` (gateway colocated with vLLM):
   direct-vs-gateway across concurrency × length distributions; overload scenario
   (arrival 2–5× capacity) comparing tail latency, error semantics, recovery;
   ≥1 profile-driven optimization with before/after numbers.
5. `gateway/README.md`: architecture diagram, admission policy spec, non-goals
   [D7], benchmark methodology + results.

**Gate**
- [ ] ASan/UBSan/TSan green  - [ ] failure-injection suite green on mock upstream
- [ ] overload = bounded queue + fast failure, never unbounded growth
- [ ] direct-vs-gateway overhead quantified  - [ ] one profiled optimization documented
- [ ] resume-claim sentence drafted from measured numbers

**Constraints**: non-goals in [D7] are hard scope walls. This thread never touches
`src/forge/` training/eval files (parallel-safety with Phases 3–4).

---

## Phase 6 — Cascade integration + release (local, 3–5 days)

**Goal**: close the loop with nlp-eval-lab and publish.

**Deliverables**
1. PR to `~/nlp-eval-lab`: add the forged model (measured quality/latency/cost from
   Phase 4) as a cascade tier; re-optimize thresholds on its CAL split; update its
   cost-model headline. Obey that repo's CLAUDE.md hard rules (append-only runs,
   frozen splits).
2. This repo's final README: headline with CIs, ladder table, serving table,
   "production story" section (what broke, what was measured, what was traded off),
   negative results, Model Card, limitations.
3. `demo/`: offline-first static site (style-match the two sister labs) — ladder
   explorer, serving dashboards, constraint-tax exhibit.
4. `make reproduce-headline`: replays the claim chain from raw artifacts (no new
   GPU/API calls), hash-gated like nlp-eval-lab's equivalent.

**Gate**
- [ ] fresh-clone `SMOKE=1` full chain green  - [ ] every README claim traceable
- [ ] `make reproduce-headline` green  - [ ] demo built  - [ ] cascade PR opened

---

## Milestones

| Week | Content |
|---|---|
| W1 | Phase 0 + Phase 1 |
| W2 | Phase 2; Phase 3 starts (R0–R1); Phase 5 thread starts |
| W3–W4 | Phase 3 completes (R2–R4); Phase 5 continues |
| W5 | Phase 4; Phase 5 feature-freeze |
| W6 | Phase 5 remote bench; Phase 6 release |

Resume checkpoint 1 (end W4): ladder R1–R3 + first vLLM numbers — post-training
story stands. Resume checkpoint 2 (end W6): gateway + cascade — systems story stands.
