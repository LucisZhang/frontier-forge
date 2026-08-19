---
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3.5
- structured-output
- tool-calling
- post-training
- gptq
- speculative-decoding
base_model: Qwen/Qwen3.5-4B-Base
---

# Frontier Forge R1b

Frontier Forge R1b is a Qwen3.5-4B-Base derivative post-trained to turn a CFPB
consumer-complaint narrative plus source metadata into a strict structured triage
ticket and one tool call. It is a research artifact for reproducible post-training,
serving, and gateway experiments—not a consumer-decision system.

## Model variants

| Repository subfolder | Contents | Tree SHA-256 |
|---|---|---|
| `bf16/` | PEFT merge-and-unload, BF16 | `7cf43a2905513f61797b78b7e3fd7ebdacd1cba4fc89abea9ce209401e6e6435` |
| `gptq-int4/` | deployment GPTQ-int4, group size 128, 128 calibration rows | `c99b42cf0e062cc75f2df8588725d0c29383666f3db0c1ae837ce15bfe6d39d2` |
| `bf16-mtp-preserved/` | merged BF16 plus 15 byte-preserved native `mtp.*` tensors | `7878b55f6fe6a9ecb12b9504b1a88d7bc6fef7ba72d91289b6e8d694f6bc75ce` |

The source adapter was trained with NF4 QLoRA. GPTQ is a separate
deployment-time quantization step; it is not the training representation. The MTP
variant copies the 15 native MTP tensors from the fixed base revision because the
language-model-only adapter does not modify them.

Base checkpoint: `Qwen/Qwen3.5-4B-Base` at revision
`1001bb4d826a52d1f399e183466143f4da7b741b`.

## Intended task and input contract

The input contract is JSON with five fields:

```json
{
  "complaint_id": 123,
  "narrative": "consumer complaint text",
  "source_product": "Mortgage",
  "source_issue": "Trouble during payment process",
  "source_company": "Example Company"
}
```

The output is strict JSON containing normalized `product`, `issue`, `company`,
`urgency`, `ambiguity_flag`, and exactly one `tool_call`. Allowed tools are
`request_more_info`, `close_no_action`, `escalate_to_regulator`,
`start_refund_workflow`, and `route_to_company`.

Source metadata is intentionally model-visible. This matches a triage workflow in
which upstream intake metadata already exists, but it means the model must **not**
be presented as a leakage-free narrative-only product classifier.

## Training

- Base: Qwen3.5-4B-Base, pretrained-only checkpoint.
- Rung: R1b, one epoch of language-model-only QLoRA SFT.
- Data: 20,000 deterministic rule-labeled TRAIN examples, disjoint from frozen
  TEST-IID and TEST-DRIFT under a 13-token contamination audit.
- LoRA: rank 16, alpha 32, dropout 0; attention and MLP projection targets; visual
  subtree excluded.
- Full-run precision: NF4 QLoRA training, BF16 evaluation.
- Hardware/cost: 15.2358 RTX 4090 GPU-hours at the owner-supplied $0.30/hour rate,
  totaling $4.57075.
- Dataset hash:
  `d1092cd16f604c25ed6d5034bf0fed33afbe975af4048fbd2a0cd93d7b25e564`.

## Evaluation

Frozen evaluation uses 1,000 TEST-IID and 1,000 TEST-DRIFT rows selected with a
fixed seed. The primary metric is a hard AND over urgency, ambiguity, tool choice,
and structural tool-argument validity. Product/issue/company normalization is
secondary and excluded from task success and reward.

| Metric | Result |
|---|---:|
| Task success | **99.05%** |
| 95% bootstrap CI | **[98.60%, 99.45%]** |
| Schema validity | 100.00% |
| Tool accuracy | 99.15% |
| Paired gain over 1,450-row R1 | **+32.70 pp [30.60, 34.50]** |

Intervals use 1,000 fixed-seed bootstrap resamples over 2,000 frozen examples. The
ground truth is the versioned rule policy, not independent human semantic labels.

## Serving measurements

On one RTX 4090 with vLLM 0.17.0, native MTP, 4 QPS fixed-seed Poisson arrivals,
and 20 measured requests, `bf16-mtp-preserved/` recorded:

- client E2E p50/p95: 1.063 / 1.311 seconds;
- output throughput: 320.6 tokens/second;
- verifier success: 95% (19/20);
- peak device memory: 21,587 MiB;
- cost: $0.0202 per 1,000 verifier-successful tasks at $0.30/GPU-hour.

Native MTP lost the matched p95/throughput gate at 0.25 QPS and won at 0.50,
1, 2, and 4 QPS, with 95.6–96.4% acceptance. These small per-cell serving samples
characterize an operating boundary; they do not replace the larger frozen quality
evaluation.

The optional C++ gateway measured 0.3% median p50 and 0.5% median p95 overhead
across five stable cells. Its overload run has a known defect: errors were admitted
requests returning HTTP 502/`upstream_error`, not designed 429 admission rejects.
Non-stable gateway cells had 10–85% errors versus 0% for bare vLLM. Lower
success-only p95 in those cells is survivor-biased and is not a model or gateway win.

## Uses

Appropriate uses:

- reproducing the documented structured-output evaluation;
- studying rule-label scaling, quantized deployment, native MTP, or constrained
  generation;
- development behind a human-reviewed complaint-triage workflow;
- research on LLM-aware gateways, after respecting the documented overload defect.

Out-of-scope uses:

- autonomous regulatory, credit, legal, refund, or account actions;
- narrative-only product classification claims;
- deployment outside the tested English CFPB taxonomy without new evaluation;
- handling live personal data without a separate privacy/security review.

## Limitations and risks

- A 200-row stratified human audit found a 14% wrong-label rate in the rule policy,
  including escalation and refund false negatives. Evaluation measures fidelity to
  those rules, so a high score can faithfully reproduce a flawed policy.
- Keyword rules are negation-blind. The single-action taxonomy also prioritizes
  escalation over refund when both triggers appear.
- R2 distilled SFT lost 14.2 points to R1; this is a task-specific result where
  executable labels are free, not a general rejection of distillation.
- The original GRPO runs were invalidated by completion parsing. R4 v2 completed
  two seeds with a +0.25-point paired delta whose CI included zero; seed 2 hit the
  locked zero-reward-variance guard. No three-seed aggregate exists.
- The TRL/Unsloth agreement gate failed, so TRL is the only reference backend.
- No safety, fairness, privacy, memorization, red-team, or human-impact evaluation
  establishes readiness for autonomous use. Complaint narratives may contain
  sensitive consumer information.
- Serving results are specific to one RTX 4090, vLLM 0.17.0, the declared workload,
  and the measurement-side definitions in the repository.

## Reproduction and provenance

The source repository ships append-only run records, raw serving receipts, failed
attempts, export manifests, and a deterministic release chain:

```bash
uv sync --locked
make reproduce-headline
make demo-build
```

The reproduction command makes no new API or GPU call. It rebuilds release payloads
from hash-pinned raw artifacts and verifies every publication file by SHA-256.

Key evidence in the source repository:

- `results/runs.jsonl`
- `results/phase3_paired_deltas.json`
- `results/phase3_export_manifest_r1b_trl_s0.json`
- `results/phase4/r1b_mtp_reexport_manifest.json`
- `results/phase4_serving_report.md`
- `results/phase5_gateway_report.md`
- `results/phase6/source_manifest.json`

## Citation

If you use the artifact, cite the source repository, the exact Hugging Face commit,
the chosen subfolder, and its tree SHA-256. The archive receipt records the remote
commit and per-file verification result after publication.
