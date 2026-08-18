# DECISIONS.md — locked decisions

These decisions were made upstream (JD gap analysis across 1,362 China campus-hire
tech JDs + three independent deep-research reports, 2026-08). Agents implement them;
they do not re-litigate them. If a decision proves infeasible during execution,
STOP and report evidence — do not improvise a substitute.

## D1. Base model: Qwen 4B-class, base (pretrained-only) checkpoint
Latest available at kickoff (reference: Qwen3.5-4B-Base). vLLM compatibility is a
hard requirement. Base — not instruct — so the before/after effect of post-training
is legible. Draft model for speculative decoding: same family, 0.5B.

## D2. Teacher: frontier API via OpenRouter
Reuse nlp-eval-lab's Tier-C setup and account. Teacher identity matters less than
data provenance: every distilled sample carries teacher model version + prompt hash.

## D3. Task: CFPB complaint → structured ticket + tool call
Input: complaint narrative. Output: strict JSON (ticket fields + one tool call from
a fixed tool registry, or abstention). Why: verifier is pure rules (schema validity,
field match, tool correctness, correct abstention) → GRPO reward comes free; reuses
nlp-eval-lab's frozen CFPB snapshot; the trained model plugs back into its cascade.
- Difficulty knobs: field count, ambiguous-sample ratio, tool count, distractor
  tools, bilingual instructions (stretch).
- Calibration target: base model zero-shot task success in the 20–50% band.
- Fallback (human decision only): executable text-to-SQL if CFPB proves
  uncalibratable. Agents report calibration data; the human decides any switch.

### D3.1 Amendment (2026-08, human-approved after Phase 1 calibration audit)
Phase 1 calibration exposed structural scoring flaws (gold `issue`/`company` copied
from metadata columns not present in the redacted narrative; hard-AND task_success;
verbatim-template tool arguments). Human decision — **input contract v2**:
- Model input = complaint narrative + source metadata fields (product/issue/company
  columns), mirroring real triage where ticket metadata exists.
- `task_success` scores DECISION fields only: urgency, ambiguity_flag, tool choice,
  structural tool-argument validity. Free-text tool arguments (`question`, `reason`)
  are scored for structure/semantics, never verbatim equality.
- `issue`/`company` normalization becomes a separately-reported secondary metric,
  excluded from task_success and from any RL reward.
- **Fair-baseline principle**: every zero-shot baseline (R0, teacher) receives the
  full task specification — urgency policy, ambiguity definition, tool registry
  semantics — in its prompt. A headline may claim execution/policy-following
  superiority only against baselines given the same spec.
- Split MEMBERSHIP stays frozen [D10]; label DERIVATION is versioned — re-derivation
  with a label_rules version bump and new dataset_hash is allowed until Phase 2 data
  generation begins, after which labels freeze too.

## D4. Training stack: TRL as reference, Unsloth as fast path
Smoke and first full run on TRL; switch to Unsloth for speed/VRAM after one
cross-check run confirms metric agreement. LLaMA-Factory optional, not required.

## D5. Experiment ladder (each rung answers one question)
R0 base zero-shot → R1 QLoRA SFT (rule-labeled data) → R2 QLoRA SFT (distilled
data) → R3 = R2 + DPO → R4 = R3 + GRPO (rule-based reward). Frozen eval set,
bootstrap CIs, paired deltas, cost per run. A negative result at any rung (e.g.
GRPO loses to DPO) is a valid, reportable outcome.

### D5.1 Amendment (2026-08-18, human-approved after two R4 aborts)
R4's original design trained GRPO on the same 1,450-row Phase 2 corpus its R3
parent had memorized (SFT+DPO), saturating every reward at 1.0 → zero advantage
→ guard abort. This is an experimental-design defect, not an algorithmic result.
**R4 v2**: GRPO from R3 on a FRESH prompt pool — ~8k TRAIN rows disjoint from all
previously trained rows (1,450 Phase 2 + 18,550 R1b additions), contamination-
screened with the Phase 2 auditor; frozen eval unchanged; num_generations 4→8.
Both prior R4 attempts remain in the ledger as superseded (bug; saturation).
Whatever v2 yields — win, loss, or tie vs R3 — is final and reportable.

## D6. Hardware: RTX 4090 24GB rental as the workhorse
AutoDL (~¥1.6–2.6/h) or Vast.ai/RunPod (~$0.35–0.65/h). Escalate to 48GB
(L40S/A6000) only if GRPO memory-pressure is demonstrated. No A100/H100, no
multi-GPU. Budget envelope: ~60–90 GPU-hours total; teacher API $20–50.
Local Mac (16GB Apple Silicon) does everything non-CUDA; every pipeline has SMOKE=1.

## D7. Gateway: LLM-aware admission control, not a generic proxy
C++20 coroutine async I/O in front of vLLM: token-aware admission (queue / degrade-
route / fast-reject), bounded queues, load shedding, circuit breaking, SSE streaming
passthrough, deadline/cancellation propagation, Prometheus metrics. Non-goals:
no tokenizer, no hand-rolled HTTP stack, no GPU scheduling, no Envoy-generality.
The resume claim is an admission policy with measured overload behavior, not LOC.

## D8. Serving metrics vocabulary
TTFT, inter-token latency, p50/p95 end-to-end, tokens/s, req/s, VRAM, max stable
concurrency, and cost-per-1k-successful-tasks (never bare tokens/s). Full disclosure
of hardware, precision, load distribution, arrival rate, warm-up, measurement side.
Speculative decoding is reported as a win/lose boundary across QPS, not a checkbox.

## D9. Structured-output deep dive is a first-class deliverable
xGrammar vs Outlines constraint overhead; reproduce tool-calling suppression under
simultaneous schema constraint + tool choice ("constraint tax"); implement two-pass
decoupling mitigation with before/after numbers.

## D10. Narrative red lines (inherited from nlp-eval-lab)
Frozen datasets/splits after first materialization; append-only results/runs.jsonl
keyed by git SHA + config hash + dataset hash; every claim traceable; no fabricated
scale; SFT-time 4-bit vs deployment quantization reported separately.
