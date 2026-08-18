# Phase 4 serving and inference engineering report

## Disclosure

- Mode: FULL GPU.
- Hardware: NVIDIA GeForce RTX 4090; driver 580.105.08; device memory 24564 MiB.
- CPU co-tenancy: the pod was shared with an unrelated CPU-only task; host has 128 logical cores. Every new spec-decode and structured sweep sampled load average and CPU utilization; any load1 sample above half the core count contaminated and reran the entire sweep. Contaminated attempts remain linked from the raw receipt.
- Historical serving sweeps completed before the CPU co-tenancy sampling requirement and are identified as such; matched spec baseline/native-MTP and both structured backends use the new clean-sweep gate.
- Serving variants and precision: r1b / bfloat16, r1b / gptq_int4, r3_equivalent_legacy_r4_zero_update / bfloat16, r3_equivalent_legacy_r4_zero_update / gptq_int4.
- vLLM version: 0.17.0.
- Workload source: frozen `mixed` evaluation rows; workload SHA-256 `4f042b56aacd6e596e112e290511717bd84737d805df67acbfbafd0845865e23`.
- Input lengths: targets [800, 1400, 2000] with weights [0.4, 0.4, 0.2]; measurement = tokenizer.apply_chat_template exact token ids.
- Output controls: max-token caps [192, 256] with weights [0.5, 0.5].
- Arrival process: Poisson, fixed seed 20260818; offered sweep [0.25, 0.5, 1.0, 2.0, 4.0] QPS; concurrency cap 32.
- Warm-up: 6 requests excluded; measurement: 20 requests per QPS point.
- Timing sides: client = monotonic wall clock around streamed OpenAI HTTP request; server = delta of vLLM Prometheus histograms over each measurement point.
- Verifier: `forge.verify.verifier.score`; input normalization = forge.train.grpo._completion_text: keep the suffix after the final </think>, then strip; request artifacts preserve both raw and normalized outputs.
- Stable means all declared error-rate, deadline-miss, achieved-QPS, and p95-inflation checks pass; max stable concurrency is the largest observed in-flight count among such points.
- Cost per 1k successful tasks uses verifier-passing requests in the denominator, never token count.
- Cache containment exception: the earliest R1b BF16 trial created `/root/.cache/vllm` before redirection was detected. The owned server was stopped, that out-of-repo cache was not modified or deleted, and all later Phase 4 caches were repo-local. Evidence: `results/phase4/logs/phase4_serve_r1b_bf16.server.log`.
- Superseded measurement retained: `phase4_serve_r1b_bf16` at `results/phase4/raw/phase4_serve_r1b_bf16.json`; reason: The original harness passed the raw </think>-prefixed completion directly to the verifier instead of applying the locked Phase 3 completion normalization.
- Superseded measurement retained: `phase4_spec_decode_r1b_bf16_baseline` at `results/phase4/raw/phase4_spec_decode_r1b_bf16_baseline.json`; reason: matched baseline rerun on the MTP-preserving target with CPU co-tenancy sampling

## Serving sweep

Client timings are wall-clock observations around the streamed HTTP request. ITL is the per-request mean `(E2E - TTFT) / (completion_tokens - 1)`. The server-side table below is independently derived from vLLM Prometheus histograms.

| Variant | Precision | QPS | TTFT p50 s | ITL p50 s | E2E p50 s | E2E p95 s | tok/s | req/s | Success | VRAM peak MiB | Cost / 1k success | Stable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1b | bfloat16 | 0.250 | 0.1182 | 0.0100 | 0.8461 | 0.9449 | 21.14 | 0.29 | 100.0% | 22829 | $0.2866 | yes |
| r1b | bfloat16 | 0.500 | 0.1183 | 0.0104 | 0.8844 | 1.0164 | 45.89 | 0.62 | 100.0% | 22829 | $0.1347 | yes |
| r1b | bfloat16 | 1.000 | 0.0961 | 0.0103 | 0.8487 | 0.9665 | 50.51 | 0.69 | 100.0% | 22829 | $0.1207 | no |
| r1b | bfloat16 | 2.000 | 0.1248 | 0.0122 | 1.0657 | 1.3667 | 109.13 | 1.43 | 95.0% | 22829 | $0.0615 | no |
| r1b | bfloat16 | 4.000 | 0.1746 | 0.0164 | 1.4245 | 1.6898 | 306.99 | 4.16 | 95.0% | 22829 | $0.0211 | yes |
| r1b | gptq_int4 | 0.250 | 0.1141 | 0.0045 | 0.4380 | 0.5530 | 21.25 | 0.29 | 100.0% | 22591 | $0.2851 | yes |
| r1b | gptq_int4 | 0.500 | 0.1148 | 0.0045 | 0.4339 | 0.5716 | 46.63 | 0.63 | 100.0% | 22591 | $0.1325 | yes |
| r1b | gptq_int4 | 1.000 | 0.0938 | 0.0045 | 0.4249 | 0.4634 | 51.08 | 0.70 | 100.0% | 22591 | $0.1193 | no |
| r1b | gptq_int4 | 2.000 | 0.1108 | 0.0057 | 0.5439 | 0.7587 | 113.03 | 1.48 | 95.0% | 22591 | $0.0594 | no |
| r1b | gptq_int4 | 4.000 | 0.1532 | 0.0085 | 0.8091 | 0.9628 | 335.61 | 4.54 | 95.0% | 22591 | $0.0193 | yes |
| r3_equivalent_legacy_r4_zero_update | bfloat16 | 0.250 | 0.1188 | 0.0101 | 0.8883 | 1.3069 | 23.83 | 0.29 | 50.0% | 22829 | $0.5773 | yes |
| r3_equivalent_legacy_r4_zero_update | bfloat16 | 0.500 | 0.1192 | 0.0103 | 0.9330 | 1.4420 | 51.21 | 0.61 | 45.0% | 22829 | $0.3014 | yes |
| r3_equivalent_legacy_r4_zero_update | bfloat16 | 1.000 | 0.0957 | 0.0103 | 0.8515 | 1.2724 | 54.85 | 0.69 | 55.0% | 22829 | $0.2195 | no |
| r3_equivalent_legacy_r4_zero_update | bfloat16 | 2.000 | 0.1226 | 0.0125 | 1.1538 | 1.6567 | 119.08 | 1.40 | 75.0% | 22829 | $0.0791 | no |
| r3_equivalent_legacy_r4_zero_update | bfloat16 | 4.000 | 0.1753 | 0.0164 | 1.4421 | 2.3825 | 345.88 | 4.15 | 50.0% | 22829 | $0.0402 | yes |
| r3_equivalent_legacy_r4_zero_update | gptq_int4 | 0.250 | 0.1139 | 0.0045 | 0.4436 | 0.6012 | 22.92 | 0.29 | 55.0% | 23055 | $0.5184 | yes |
| r3_equivalent_legacy_r4_zero_update | gptq_int4 | 0.500 | 0.1132 | 0.0045 | 0.4419 | 0.6747 | 51.16 | 0.63 | 50.0% | 23055 | $0.2652 | yes |
| r3_equivalent_legacy_r4_zero_update | gptq_int4 | 1.000 | 0.0930 | 0.0045 | 0.4272 | 0.6292 | 55.41 | 0.70 | 55.0% | 23055 | $0.2170 | no |
| r3_equivalent_legacy_r4_zero_update | gptq_int4 | 2.000 | 0.1101 | 0.0057 | 0.6087 | 0.8424 | 126.11 | 1.46 | 70.0% | 23055 | $0.0814 | no |
| r3_equivalent_legacy_r4_zero_update | gptq_int4 | 4.000 | 0.1477 | 0.0084 | 0.8183 | 1.1168 | 356.92 | 4.54 | 55.0% | 23055 | $0.0333 | yes |

| Variant | Precision | Max stable observed concurrency | Max stable offered QPS |
|---|---|---:|---:|
| r1b | bfloat16 | 10 | 4.000 |
| r1b | gptq_int4 | 7 | 4.000 |
| r3_equivalent_legacy_r4_zero_update | bfloat16 | 12 | 4.000 |
| r3_equivalent_legacy_r4_zero_update | gptq_int4 | 7 | 4.000 |

### Independent server-side timing

The representative point is the highest stable QPS, or the highest tested QPS when no point passes the declared stability criteria.

| Variant | Precision | QPS | Server TTFT p50/p95 s | Server ITL p50/p95 s | Server E2E p50/p95 s |
|---|---|---:|---:|---:|---:|
| r1b | bfloat16 | 4.000 | 0.1800/0.4167 | n/a/n/a | 1.3636/1.9286 |
| r1b | gptq_int4 | 4.000 | 0.1706/0.2500 | n/a/n/a | 0.8000/0.9800 |
| r3_equivalent_legacy_r4_zero_update | bfloat16 | 4.000 | 0.1800/0.4167 | n/a/n/a | 1.4091/2.5000 |
| r3_equivalent_legacy_r4_zero_update | gptq_int4 | 4.000 | 0.1618/0.2412 | n/a/n/a | 0.8222/1.2500 |

## Speculative decoding boundary

A point is a win only when speculative decoding has no worse client p95 E2E and no lower verifier-successful request throughput than the matched baseline. Boundary: **observed transition interval(s): 0.25 (lose) to 0.5 (win)**.

Method run: **mtp** (`model-native MTP`, 1 draft token). D1.3 selected the model-native path because `fixed_revision_base_index_contains_mtp_and_r1b_adapter_does_not_modify_mtp`; the fixed-revision base index contains 15 `mtp.*` weights and the R1b adapter contains none. The sibling R1b export restores those exact base tensor bytes. vLLM stays at `0.17.0` with no M-RoPE patch and no version change.

Prior failed attempts remain archived: `results/phase4/phase4_spec_decode_r1b_bf16_qwen05b_failure.json`, `results/phase4/phase4_spec_decode_r1b_bf16_qwen08b_failure.json`, `results/phase4/phase4_spec_decode_r1b_bf16_qwen08b_patched_failure.json`.

![Speculative decoding boundary](phase4_spec_decode_boundary.svg)

| QPS | Baseline p95 s | Spec p95 s | Delta s | Baseline success req/s | Spec success req/s | Acceptance | Mean acceptance length | Load1 max | CPU mean | Clean | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.250 | 0.9694 | 1.0208 | 0.0514 | 0.29 | 0.29 | 96.4% | 1.96 | 7.21 | 5.9% | yes | lose |
| 0.500 | 1.0154 | 0.7866 | -0.2288 | 0.62 | 0.62 | 96.2% | 1.96 | 6.41 | 5.3% | yes | win |
| 1.000 | 0.9652 | 0.7385 | -0.2268 | 0.69 | 0.69 | 96.0% | 1.96 | 7.01 | 5.6% | yes | win |
| 2.000 | 1.3765 | 1.0556 | -0.3208 | 1.36 | 1.38 | 95.6% | 1.96 | 7.25 | 7.6% | yes | win |
| 4.000 | 1.6817 | 1.3112 | -0.3705 | 3.95 | 4.12 | 95.6% | 1.96 | 7.35 | 8.3% | yes | win |

## Structured-output deep dive

### Backend overhead and cold compile

Cold compile overhead is first request latency minus the median repeated latency for the same previously unseen schema. Steady constraint overhead is repeated constrained median minus an unconstrained control with the same prompt. Stop rate is also shown so max-token truncation is not mistaken for a valid completion.

| Backend | Required fields | Cold finish | Steady stop rate | Cold latency s | Steady p50 s | Cold compile overhead s | Steady constraint overhead s |
|---|---:|---|---:|---:|---:|---:|---:|
| outlines | 2 | stop | 100.0% | 12.0127 | 0.2916 | 11.7212 | -0.0082 |
| outlines | 6 | stop | 100.0% | 1.5118 | 0.7303 | 0.7815 | 0.0314 |
| outlines | 12 | stop | 100.0% | 2.8675 | 1.3979 | 1.4696 | 0.0876 |
| outlines | 24 | length | 0.0% | 5.2991 | 2.6082 | 2.6908 | 0.1026 |
| xgrammar | 2 | stop | 100.0% | 11.7205 | 0.3084 | 11.4121 | -0.0060 |
| xgrammar | 6 | stop | 100.0% | 0.7492 | 0.7427 | 0.0064 | 0.0327 |
| xgrammar | 12 | stop | 100.0% | 1.4176 | 1.4103 | 0.0073 | 0.0924 |
| xgrammar | 24 | length | 0.0% | 2.6251 | 2.6111 | 0.0140 | 0.1069 |

### Constraint tax and two-pass mitigation

The one-pass condition sends `tool_choice=required` and a simultaneous response JSON schema. Two-pass first obtains a model-selected tool, then constrains the complete ticket with that selected tool fixed. Missing first-pass choices are not filled from gold labels and count as failures.

| Backend | Tools-only call rate | Simultaneous call rate | Constraint-tax delta | One-pass task success | Two-pass task success | Mitigation delta | One-pass p50/p95 s | Two-pass p50/p95 s | p50 latency delta s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| outlines | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.5446/0.5863 | 1.4247/11.2754 | 0.8801 |
| xgrammar | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.5146/0.5535 | 1.3259/1.4486 | 0.8113 |

Observed reproduction outcome:

- `outlines`: tool-call-rate tax not reproduced (both conditions retained the tool call); simultaneous one-pass correctness tax observed at 0.0% task success; two-pass task success 100.0% with p50/p95 latency 1.4247/11.2754 s.
- `xgrammar`: tool-call-rate tax not reproduced (both conditions retained the tool call); simultaneous one-pass correctness tax observed at 0.0% task success; two-pass task success 100.0% with p50/p95 latency 1.3259/1.4486 s.

### CPU co-tenancy measurements

| Backend | Logical cores | Load1 threshold | Load1 mean/max | CPU mean/max | Samples | Clean | Contaminated attempts retained |
|---|---:|---:|---:|---:|---:|---|---:|
| outlines | 128 | 64.00 | 7.96/8.32 | 6.2%/11.0% | 661 | yes | 0 |
| xgrammar | 128 | 64.00 | 8.25/10.36 | 6.5%/9.5% | 450 | yes | 0 |

## Phase 4 gate

- [x] Disclosure block includes hardware, precision, load distribution, arrival rate, warm-up, and timing side.
- [x] New latency-sensitive sweeps record CPU/load co-tenancy and final headline attempts are below the contamination threshold.
- [x] Cost per 1k successful tasks is computed against verifier-passing requests.
- [x] Speculative-decoding acceptance rate is recorded at every QPS point.
- [x] Speculative-decoding win/lose boundary is tabulated and plotted.
- [x] Constraint-tax tool-call rates and two-pass task-success/latency deltas are reported.
- [x] Report and SVG are deterministic functions of hash-checked raw artifacts.

## Reproduction

```bash
make bench-report
```

Raw provenance:

- `phase4_serve_r1b_bf16_v2`: `results/phase4/raw/phase4_serve_r1b_bf16_v2.json`; config `configs/phase4/serve_r1b_bf16.yaml`; Git `e1150dc39384141dd25c8b52796c1bffaa730c53`.
- `phase4_serve_r1b_gptq_int4`: `results/phase4/raw/phase4_serve_r1b_gptq_int4.json`; config `configs/phase4/serve_r1b_gptq_int4.yaml`; Git `e1150dc39384141dd25c8b52796c1bffaa730c53`.
- `phase4_serve_r3eq_bf16`: `results/phase4/raw/phase4_serve_r3eq_bf16.json`; config `configs/phase4/serve_r3eq_bf16.yaml`; Git `e1150dc39384141dd25c8b52796c1bffaa730c53`.
- `phase4_serve_r3eq_gptq_int4`: `results/phase4/raw/phase4_serve_r3eq_gptq_int4.json`; config `configs/phase4/serve_r3eq_gptq_int4.yaml`; Git `e1150dc39384141dd25c8b52796c1bffaa730c53`.
- `phase4_spec_decode_r1b_bf16_baseline_v2`: `results/phase4/raw/phase4_spec_decode_r1b_bf16_baseline_v2.json`; config `configs/phase4/spec_r1b_bf16_baseline.yaml`; Git `db50ccb516528fec62b2dfa9ca82817cc477db46`.
- `phase4_spec_decode_r1b_bf16_native_mtp`: `results/phase4/raw/phase4_spec_decode_r1b_bf16_native_mtp.json`; config `configs/phase4/spec_r1b_bf16_mtp.yaml`; Git `db50ccb516528fec62b2dfa9ca82817cc477db46`.
- `phase4_structured_r1b_bf16_outlines`: `results/phase4/raw/phase4_structured_r1b_bf16_outlines.json`; config `configs/phase4/structured_r1b_bf16_outlines.yaml`; Git `db50ccb516528fec62b2dfa9ca82817cc477db46`.
- `phase4_structured_r1b_bf16_xgrammar`: `results/phase4/raw/phase4_structured_r1b_bf16_xgrammar.json`; config `configs/phase4/structured_r1b_bf16_xgrammar.yaml`; Git `db50ccb516528fec62b2dfa9ca82817cc477db46`.
