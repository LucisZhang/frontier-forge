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

## D4. Training stack: TRL as reference, Unsloth as fast path
Smoke and first full run on TRL; switch to Unsloth for speed/VRAM after one
cross-check run confirms metric agreement. LLaMA-Factory optional, not required.

## D5. Experiment ladder (each rung answers one question)
R0 base zero-shot → R1 QLoRA SFT (rule-labeled data) → R2 QLoRA SFT (distilled
data) → R3 = R2 + DPO → R4 = R3 + GRPO (rule-based reward). Frozen eval set,
bootstrap CIs, paired deltas, cost per run. A negative result at any rung (e.g.
GRPO loses to DPO) is a valid, reportable outcome.

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
