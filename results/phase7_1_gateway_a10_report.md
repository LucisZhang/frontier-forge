# Phase 7.1 A10 gateway matched rerun report

Status: **Gate 7.1 FAIL**. Run `phase7_1_gateway_r1b_bf16_native_mtp_a10` at git `14f03be64000efc539944fce91d9cb77153b1fa4`. Raw receipt: `results/phase7_1/raw/phase7_1_gateway_bench.json`.

## Decision

Production remains blocked. The README historical red-flag section must remain unchanged.

The rerun used one **NVIDIA A10** (23028 MiB) on the Aliyun ECS VM. Phase 5 used an RTX 4090. The 4090 evidence below is retained only as defect history; **no A10 latency or throughput number is compared with a 4090 number**.

All A10 bare-vLLM capacity, nine-cell concurrency × length matrix, and overload baseline cells completed before the gateway process started. The same A10 model artifact, BF16 precision, vLLM 0.17.0, request rows, seeds, warm-ups, schedules, measurement counts, and deadlines were then used for the gateway cells.

## Archived Phase 5 finding (RTX 4090; history retained)

| RTX 4090 Phase 5 load | bare errors | gateway errors | observed semantics |
|---:|---:|---:|---|
| 2× | 0/60 | 8/60 | HTTP statuses {'200': 52, '502': 8}; codes upstream_error:8 |
| 3× | 0/60 | 7/60 | HTTP statuses {'200': 53, '502': 7}; codes upstream_error:7 |
| 5× | 0/60 | 14/60 | HTTP statuses {'200': 46, '502': 14}; codes upstream_error:14 |

Every archived error above passed admission and returned HTTP 502/`upstream_error`; `reject_overload=0`. Those cells are not relabeled as 429 and their success-only latency remains survivor-biased.

## Phase 7.1 A10 overload rerun

| 负载 | bare A10 5xx | gateway A10 5xx | gateway 429 | reject_overload | queue sampled/process | 429 p50 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2× | 0/60 | 0/60 | 0/60 | 0 | 8/8 | n/a |
| 3× | 0/60 | 0/60 | 0/60 | 0 | 20/20 | n/a |
| 5× | 0/60 | 0/60 | 14/60 | 14 | 24/24 | 1.3 |

The gate defines upstream-5xx parity before looking at results: each gateway cell must be within **±5.0 percentage points** of its paired A10 bare-vLLM cell. The same band also applies to all non-admission errors, so transport failures without an HTTP status cannot disappear from the gate. Every overload cell must contain at least 1 HTTP 429 response, with `Retry-After`, error code `overloaded`, client latency ≤1.0 s, and both sampled and process queue high-watermarks ≤24.

## Same-box A10 matched matrix

| 长度 | 并发 | bare/gateway errors | E2E p50 overhead | E2E p95 overhead | throughput delta | interpretation |
|---|---:|---:|---:|---:|---:|---|
| short | 1 | 0.0%/0.0% | -0.5% | -0.4% | 0.5% | paired A10 stable cell |
| short | 8 | 0.0%/0.0% | -0.5% | -0.6% | 0.4% | paired A10 stable cell |
| short | 32 | 0.0%/0.0% | -7.3% | -6.6% | 6.2% | paired A10 stable cell |
| mixed | 1 | 0.0%/0.0% | -0.5% | -0.4% | 0.5% | paired A10 stable cell |
| mixed | 8 | 0.0%/0.0% | -4.3% | 4.6% | 1.0% | paired A10 stable cell |
| mixed | 32 | 0.0%/0.0% | -3.5% | 12.4% | -11.9% | paired A10 stable cell |
| long | 1 | 0.0%/0.0% | -0.3% | -0.3% | 0.2% | paired A10 stable cell |
| long | 8 | 0.0%/0.0% | -1.8% | -0.3% | 1.1% | paired A10 stable cell |
| long | 32 | 0.0%/0.0% | -4.1% | 10.2% | -9.5% | paired A10 stable cell |

Across 9 stable A10 pairs, median same-box gateway overhead was p50 **-1.8%**, p95 **-0.3%**, and throughput delta **0.5%**. These claims are A10-to-A10 only.

## Cost and pricing assumption

- `FORGE_GPU_HOURLY_USD=1.53`.
- The rate assumes **¥11/hour at 7.2 CNY/USD**: 11 / 7.2 = 1.5278, rounded to $1.53/hour.
- Accounted delegated VM session: 3.3493 h = **$5.1244**. Time after final receipt while the owner stops the VM is excluded and disclosed.

## Gate 7.1

- [x] root cause documented with evidence
- [x] regression tests old-fail/new-pass
- [x] matched matrix + overload rerun receipts
- [ ] 429 semantics and bounded queue verified
- [x] upstream 5xx rate within the pinned ±5.0 pp of paired bare vLLM
- [x] all non-admission errors, including transport failures without an HTTP status, stayed within the same parity band
- [x] production-blocked disposition recorded: retained because one or more remote acceptance checks failed

## Reproduction

```sh
export FORGE_GPU_HOURLY_USD=1.53
export FORGE_BENCH_GIT_SHA=14f03be64000efc539944fce91d9cb77153b1fa4
export FORGE_VM_STARTED_AT=2026-08-20T12:21:12+00:00
python -m gateway.bench.phase7_1_bench --stage verify-artifact
python -m gateway.bench.phase7_1_bench --stage baseline
python -m gateway.bench.phase7_1_bench --stage gateway-matrix
python -m gateway.bench.phase7_1_bench --stage gateway-overload
python -m gateway.bench.phase7_1_report
```
