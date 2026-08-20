# Phase 7.1 sustained-overload amendment report

Status: **Gate 7.1 PASS**. Run `phase7_1_sustained_overload_r1b_bf16_native_mtp_a10` at git `aacacdd89de5a74deb005ea86ba16dff8f74c1cb`. Raw receipt: `results/phase7_1/raw/phase7_1_sustained_gateway_bench.json`.

## Decision

The Phase 5 production block is lifted for the measured single-node gateway overload contract only. This does not claim cloud production, multi-GPU scaling, or Kubernetes readiness.

The amendment used one same-box **NVIDIA A10** (23028 MiB) on Aliyun ECS. Each 2×/3×/5× cell used fixed-seed Poisson arrivals through at least 120 seconds. All bare-vLLM cells completed before the gateway process started; direct and gateway schedules match by SHA-256. The A10 is a hardware substitution for the archived RTX 4090, so no cross-GPU latency or throughput comparison is made.

## Before receipt: archived RTX 4090 defect

| RTX 4090 Phase 5 load | bare errors | gateway errors | observed semantics |
|---:|---:|---:|---|
| 2× | 0/60 | 8/60 | HTTP statuses {'200': 52, '502': 8}; codes upstream_error:8 |
| 3× | 0/60 | 7/60 | HTTP statuses {'200': 53, '502': 7}; codes upstream_error:7 |
| 5× | 0/60 | 14/60 | HTTP statuses {'200': 46, '502': 14}; codes upstream_error:14 |

Those 29 Phase 5 overload errors were admitted HTTP 502/`upstream_error` responses with `reject_overload=0`; success-only latency in error-bearing cells remains survivor-biased history and is not relabeled.

## Intermediate receipt: finite A10 burst

| 负载 | bare A10 5xx | gateway A10 5xx | gateway 429 | reject_overload | queue sampled/process | 429 p50 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2× | 0/60 | 0/60 | 0/60 | 0 | 8/8 | n/a |
| 3× | 0/60 | 0/60 | 0/60 | 0 | 20/20 | n/a |
| 5× | 0/60 | 0/60 | 14/60 | 14 | 24/24 | 1.3 |

The finite 60-request A10 burst had queue high-watermarks 8, 20, and 24 at 2×, 3×, and 5×. Only 5× reached the bound and correctly shed 14 requests. The original per-cell “at least one 429” proxy therefore failed for 2×/3× even though those finite bursts fit in the configured queue. This receipt remains valid; only that proxy was superseded by the human-approved duration amendment.

## After receipt: duration-based A10 overload

| load | arrival windows bare/gateway | requests bare/gateway | bare 5xx / transport | gateway 5xx / 429 | queue sampled/process | 429 p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 2× | 120.1s / 120.1s | 510 / 510 | 0 (0.0%) / 0 | 0 (0.0%) / 19 | 24/24 | 1.5 ms |
| 3× | 120.4s / 120.4s | 707 / 707 | 0 (0.0%) / 0 | 0 (0.0%) / 215 | 24/24 | 2.1 ms |
| 5× | 120.1s / 120.1s | 1177 / 1177 | 36 (3.1%) / 651 | 0 (0.0%) / 687 | 24/24 | 1.9 ms |

The sustained bare-vLLM 5× cell is also a retained negative result: after 490 HTTP 200 responses and 36 HTTP 500 responses, vLLM 0.17.0 terminated its EngineCore on `AssertionError: num_decodes: 1, num_spec_decodes: 26`; the remaining 651 requests recorded transport errors. There was no host OOM, NVIDIA Xid, or co-tenancy contamination. The matched gateway 5× cell kept vLLM alive and returned 490 HTTP 200 plus 687 bounded admission 429 responses. This failure is disclosed separately from the predeclared HTTP-5xx parity calculation; it is not hidden by the passing gate.

The gate was declared before this rerun: every direct and gateway arrival window must be at least 120 seconds; every gateway cell must visibly sample the queue at its 24-request bound; all excess admission rejects must be HTTP 429/`overloaded`, carry `Retry-After`, and complete within 1.0 second; admitted gateway 5xx rate must stay within ±5.0 percentage points of paired bare vLLM.

## Cost

- Rate: `FORGE_GPU_HOURLY_USD=1.53`, derived from ¥11/hour at 7.2 CNY/USD.
- Delegated session through this receipt: 0.6139 h = **$0.9393**.

## Gate 7.1

- [x] root cause documented with evidence
- [x] regression tests old-fail/new-pass
- [x] matched matrix + finite and sustained overload receipts
- [x] duration-based arrivals ≥120 s at 2×/3×/5× for bare vLLM and gateway
- [x] queue saturated at the configured bound without exceeding it
- [x] excess surfaced as fast HTTP 429/overloaded with Retry-After
- [x] admitted gateway upstream 5xx stayed within ±5.0 pp of paired bare vLLM
- [x] production-blocked disposition recorded: lifted for the measured single-node gateway contract

## Reproduction

```sh
export FORGE_GPU_HOURLY_USD=1.53
export FORGE_BENCH_GIT_SHA=aacacdd89de5a74deb005ea86ba16dff8f74c1cb
export FORGE_PHASE7_SESSION_STARTED_AT=2026-08-20T16:03:31+00:00
python -m gateway.bench.phase7_1_sustained --stage verify-artifact
python -m gateway.bench.phase7_1_sustained --stage bare
# start the pinned gateway only after the bare stage completes
python -m gateway.bench.phase7_1_sustained --stage gateway
python -m gateway.bench.phase7_1_sustained_report --update-readme
```
