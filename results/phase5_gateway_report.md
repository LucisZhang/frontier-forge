# Phase 5 gateway remote benchmark report

Status: **complete**. Run `phase5_gateway_r1b_bf16_native_mtp` on git `9dff2a6758a1ed2facd5f132d0f22de182a592bb`; baseline gateway git `2c5d6a4a364f13e84e999dfd78d32288d002f205`. Raw receipt: `results/phase5/raw/phase5_gateway_bench.json`.

## Result

- Bare-vLLM measured capacity: **2.000 QPS**; max stable observed concurrency **6**.
- Across 5 stable direct/gateway cells, median gateway E2E overhead was **p50 0.3% / p95 0.5%**; median throughput delta **-0.9%**.
- Profile-driven optimization: `Adaptive queued-admission polling backoff from 5 ms to 10 ms and then 20 ms while preserving immediate first admission and deadline semantics`. Matched profile cell E2E p50 changed **3.655 → 3.821 s (4.5%)**; throughput **7.752 → 8.029 req/s (3.6%)**.

## Resume claim draft

> 在单卡 RTX 4090 上为 R1b BF16 + 原生 MTP vLLM 实现 C++20 token-aware admission gateway：稳定单元格端到端 p50 中位开销 0.3%，5× 过载时队列峰值 10、HTTP 502/upstream_error 快速失败 p50 5.0 ms，恢复 4.485 s（裸 vLLM 4.649 s）。

## Capacity calibration

| offered QPS | achieved ratio | p95 E2E s | peak concurrency | error | deadline miss | stable | load1 max | CPU max |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.250 | 1.224 | 0.845 | 3 | 0.0% | 0.0% | yes | 12.69 | 24.7% |
| 0.500 | 0.748 | 0.737 | 1 | 0.0% | 0.0% | no | 16.45 | 24.8% |
| 0.750 | 0.803 | 0.995 | 3 | 0.0% | 0.0% | no | 15.29 | 26.9% |
| 1.000 | 1.384 | 0.996 | 5 | 0.0% | 0.0% | yes | 16.18 | 27.7% |
| 1.500 | 1.395 | 0.908 | 5 | 0.0% | 0.0% | yes | 15.43 | 14.9% |
| 2.000 | 1.251 | 1.004 | 6 | 0.0% | 0.0% | yes | 15.43 | 14.7% |

Capacity is the highest offered Poisson QPS that passes the pinned error, deadline, achieved-QPS, and p95-inflation gates.

## Direct vs gateway: concurrency × length distribution

| 长度分布 | 并发 | 端点 | TTFT p50/p95 ms | ITL p50/p95 ms | E2E p50/p95 s | req/s | tok/s | 成功任务成本 $/1k | 错误率 | VRAM peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | direct | 81.0/87.6 | 7.4/7.4 | 0.602/0.679 | 1.635 | 118.6 | 0.0510 | 0.0% | 21601 |
| short | 1 | gateway | 83.4/97.6 | 7.4/7.5 | 0.604/0.683 | 1.627 | 118.0 | 0.0512 | 0.0% | 21601 |
|  |  | paired overhead |  |  | p50 0.3%, p95 0.5% | throughput -0.5% |  |  |  |  |
| short | 8 | direct | 263.1/483.6 | 10.2/12.7 | 1.023/1.235 | 7.060 | 510.8 | 0.0118 | 0.0% | 21601 |
| short | 8 | gateway | 259.2/496.0 | 10.1/12.0 | 1.035/1.224 | 6.936 | 501.8 | 0.0120 | 0.0% | 21601 |
|  |  | paired overhead |  |  | p50 1.2%, p95 -0.9% | throughput -1.8% |  |  |  |  |
| short | 32 | direct | 743.1/1135.2 | 14.5/20.8 | 1.762/1.838 | 10.653 | 752.7 | 0.0078 | 0.0% | 21601 |
| short | 32 | gateway | 567.9/833.3 | 12.4/16.1 | 1.404/1.491 | 9.768 | 698.1 | 0.0085 | 25.0% | 21601 |
|  |  | paired overhead |  |  | p50 -20.3%, p95 -18.9% | throughput -8.3% |  |  |  |  |
| mixed | 1 | direct | 111.7/122.3 | 7.4/7.5 | 0.640/0.720 | 1.546 | 113.7 | 0.0539 | 0.0% | 21601 |
| mixed | 1 | gateway | 117.4/127.2 | 7.4/7.5 | 0.641/0.722 | 1.531 | 112.6 | 0.0544 | 0.0% | 21601 |
|  |  | paired overhead |  |  | p50 0.3%, p95 0.3% | throughput -0.9% |  |  |  |  |
| mixed | 8 | direct | 415.0/718.7 | 11.4/14.6 | 1.252/1.636 | 5.840 | 421.4 | 0.0143 | 0.0% | 21601 |
| mixed | 8 | gateway | 374.1/665.0 | 12.1/14.6 | 1.198/1.773 | 5.604 | 405.4 | 0.0149 | 10.0% | 21601 |
|  |  | paired overhead |  |  | p50 -4.3%, p95 8.4% | throughput -4.0% |  |  |  |  |
| mixed | 32 | direct | 1121.6/1712.5 | 17.5/23.9 | 2.361/2.443 | 8.085 | 602.8 | 0.0103 | 0.0% | 21601 |
| mixed | 32 | gateway | 472.7/644.6 | 10.4/13.1 | 1.257/1.290 | 6.813 | 510.2 | 0.0122 | 55.0% | 21601 |
|  |  | paired overhead |  |  | p50 -46.8%, p95 -47.2% | throughput -15.7% |  |  |  |  |
| long | 1 | direct | 121.3/138.2 | 7.5/7.6 | 0.670/0.753 | 1.469 | 109.8 | 0.0567 | 0.0% | 21601 |
| long | 1 | gateway | 123.6/133.6 | 7.5/7.6 | 0.672/0.761 | 1.464 | 109.5 | 0.0569 | 0.0% | 21601 |
|  |  | paired overhead |  |  | p50 0.3%, p95 1.0% | throughput -0.3% |  |  |  |  |
| long | 8 | direct | 366.5/873.6 | 13.7/16.3 | 1.367/1.952 | 5.188 | 389.1 | 0.0161 | 0.0% | 21601 |
| long | 8 | gateway | 338.3/349.7 | 7.8/8.9 | 0.880/0.883 | 3.326 | 236.2 | 0.0251 | 85.0% | 21601 |
|  |  | paired overhead |  |  | p50 -35.6%, p95 -54.8% | throughput -35.9% |  |  |  |  |
| long | 32 | direct | 1263.8/2110.0 | 19.3/29.6 | 2.720/2.861 | 6.846 | 512.7 | 0.0122 | 0.0% | 21601 |
| long | 32 | gateway | 1115.7/2430.2 | 16.4/22.5 | 2.441/3.132 | 5.954 | 445.9 | 0.0140 | 5.0% | 21601 |
|  |  | paired overhead |  |  | p50 -10.3%, p95 9.5% | throughput -13.0% |  |  |  |  |

Every direct/gateway pair used identical serialized request hashes, offsets, warm-up count, model artifact, precision, and hardware. Pair execution order alternated to reduce ordering bias.

## Overload: 2× / 3× / 5× capacity

| 倍数 | 端点 | offered QPS | E2E all p95 s | 成功 p95 s | error | fast-reject p50 ms | queue max | fallback | recovery s | 错误语义 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2× | direct | 4.000 | 1.390 | 1.390 | 0.0% | n/a | n/a | 0.0% | 2.336 | status[200:60] code[none] |
| 2× | gateway | 4.000 | 1.097 | 1.106 | 13.3% | 5.3 | 0 | 0.0% | 2.359 | status[200:52, 502:8] code[upstream_error:8] |
| 3× | direct | 6.000 | 2.112 | 2.112 | 0.0% | n/a | n/a | 0.0% | 2.516 | status[200:60] code[none] |
| 3× | gateway | 6.000 | 1.858 | 1.894 | 11.7% | 6.5 | 1 | 0.0% | 2.497 | status[200:53, 502:7] code[upstream_error:7] |
| 5× | direct | 10.000 | 3.576 | 3.576 | 0.0% | n/a | n/a | 0.0% | 4.649 | status[200:60] code[none] |
| 5× | gateway | 10.000 | 2.550 | 2.570 | 23.3% | 5.0 | 10 | 0.0% | 4.485 | status[200:46, 502:14] code[upstream_error:14] |

Fallback was deliberately disabled: this run had one physical R1b MTP vLLM replica, so routing the same backend through a second logical pool would fabricate independent fallback capacity. Fallback share is therefore honestly reported as zero.

## Profile-driven optimization

- Profiler: `GNU gprof 2.38 (-pg); matched 80-request gateway profile cells, with multithread call counts interpreted directionally`; baseline artifact `results/phase5/profile/gprof-before.txt` (`1f5c0bd3deaef9480892ff07613076e9ba6cc41864de3f65a0f979e5fee8d764`).
- Largest measured gateway-side cost: `boost::asio::detail::scheduler::do_run_one led sampled gateway self time at 16.28% (0.07 s, 50,920 calls); admission polling accounted for 26,946 calls in the matched baseline profile`.
- Change: `Adaptive queued-admission polling backoff from 5 ms to 10 ms and then 20 ms while preserving immediate first admission and deadline semantics`.
- Matched request schedule: `5947fb6de726c0794623f1cc389248e0c4d5e72e09b770b604a3801117f9ce07`.
- E2E p95: 5.074 → 4.937 s (-2.7%).

## Disclosure

- Hardware: `NVIDIA GeForce RTX 4090`, 24564 MiB VRAM, 128 logical CPUs, driver `580.105.08`.
- Model: R1b MTP-preserving export `7878b55f6fe6a9ecb12b9504b1a88d7bc6fef7ba72d91289b6e8d694f6bc75ce`; BF16 deployment, no deployment quantization. Training used QLoRA NF4; these are separate facts.
- Server: vLLM `0.17.0`, native MTP speculative decoding with 1 speculative token, max model length 4096, max sequences 32.
- Load: closed-loop fixed concurrency for overhead cells; fixed-seed Poisson arrivals for capacity and overload. Warm-up: 4 requests per cell, excluded.
- Measurement side: client monotonic streaming latency; vLLM and gateway Prometheus deltas; device-wide `nvidia-smi` VRAM; server-host load average and `/proc/stat` CPU utilization.
- Co-tenancy: The original pod was shared with an unrelated CPU-only task but had no GPU capacity, so the owner authorized a same-region clone on a separate host. The clone had no unrelated task inside its container; the provider host may still be multi-tenant. Every warm-up and measured cell sampled host load average and CPU utilization; any load1 sample above half the host logical core count contaminated and reran the entire stage. Maximum sampled load1 **18.71** vs contamination threshold **64.0** (128 host logical CPUs); max host CPU utilization **27.7%**. The container's provider allocation was 16 CPU cores; it is disclosed separately and is not mixed with the host-level load/CPU denominator. Contaminated stage attempts retained/rerun: **0**.
- Cost: `$0.30/GPU-hour`; accounted task uptime 1.1502 h = `$0.3450`. Owner-requested post-task running time is not yet in this closed receipt.
- Successful-task cost uses verifier successes, never raw token counts.

## Reproduction

```sh
FORGE_GPU_HOURLY_USD=0.30 FORGE_BENCH_GIT_SHA=<sha> \
  DIRECT_URL=http://127.0.0.1:8000 GATEWAY_URL=http://127.0.0.1:9000 \
  make gateway-bench STAGE=<capacity|profile-before|profile-after|final>
make gateway-bench-report
```

## Phase 5 gate

- [x] ASan/UBSan green
- [x] TSan green
- [x] failure-injection suite green on mock upstream
- [x] overload = bounded queue + fast failure, never unbounded growth
- [x] direct-vs-gateway overhead quantified
- [x] one profile-driven optimization documented with matched before/after requests
- [x] resume-claim sentence drafted from measured numbers
