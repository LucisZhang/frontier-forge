# Phase 7.1 gateway overload-semantics 本地报告

状态：**本地缓解、429 语义修正与回归门禁完成；远程 matched matrix 尚未运行。** 本报告只覆盖
PLAN.md Phase 7.1 的本地范围，不改写 Phase 5 的负面结果，也不解除
`production blocked`。

独立审查后，`gateway/README.md` 的 queue-full 行已从 503 更正为 429；该文件
属于 Phase 6 release manifest，因此已通过 `make phase6-release-write` 与
`make reproduce-headline` 重新封存，没有手工改写 manifest。

## 证据链

Phase 5 的源记录是 `results/runs.jsonl` 中的
`phase5_gateway_r1b_bf16_native_mtp`（测量源码 git
`9dff2a6758a1ed2facd5f132d0f22de182a592bb`），原始总回执为
`results/phase5/raw/phase5_gateway_bench.json`。其 overload 单元格记录：

| 负载 | bare vLLM | gateway | admission 决策 | queue high-watermark |
|---:|---|---|---|---:|
| 2x | 60×200 | 52×200 + 8×502 | primary=63, queued=0, reject_overload=0 | 7 |
| 3x | 60×200 | 53×200 + 7×502 | primary=63, queued=1, reject_overload=0 | 7 |
| 5x | 60×200 | 46×200 + 14×502 | primary=64, queued=17, reject_overload=0 | 11 |

29 个 overload 502 的客户端耗时范围是 2.516–9.056 ms，中位数 5.251 ms；
请求 deadline 是 60 s。可复现查询：

```sh
jq -s '[.[] | select(.http_status == 502) | .client_e2e_s] |
  {count:length,min_s:min,max_s:max,p50_s:(sort | .[length/2|floor])}' \
  results/phase5/requests/overload.requests.jsonl
```

## 根因与排除项

根因是**连接池把“本地 fd 仍为 open”误当成“对端 keep-alive 连接仍可用”**。
旧实现的 `UpstreamPool::ensure_connected` 只检查 `socket().is_open()`；上游在
空闲期发送 FIN/reset 后，本地 descriptor 仍可能保持 open。下一次已通过
admission 的 POST 因而复用半关闭连接，在 write/read-header 阶段立即得到
EOF/reset，再由 `proxy_request` 映射为 HTTP 502/`upstream_error`。Phase 5
bench 又在 direct/gateway 单元格之间交替执行，连接会跨 bare-vLLM 单元格
空闲；每格只有 4 次 warm-up，不能清理此前高并发打开的全部空闲连接。

排除结果：

- **连接池大小耗尽不是这些快速 502 的主因。** 远程池为 64，而 vLLM
  `max_num_seqs=32`；若等待 pool slot 或 60 s deadline 到期，错误不会在
  2.5–9.1 ms 内集中返回。
- **同一连接并发复用被排除。** `in_use` 在互斥锁内领取/释放，既有 bounded
  pool 并发测试继续通过。
- **deadline 与慢 stream 的竞态不是这批 502 的主因。** 失败耗时比 60 s
  deadline 小四个数量级；注入 latency 的 504 路径仍由独立测试覆盖。
- **upstream keep-alive 复用被确认。** mock 上游先返回声明可 keep-alive 的
  200，再在空闲期主动关闭。旧源码下第二次请求稳定得到 502；改动后网关
  在发送 POST 前丢弃该 socket、重连并得到 200。

由于 Phase 5 gateway 日志没有保存 502 的底层 `system_error` 文本，远程历史
回执不能单独展示每次 EOF/reset 字符串；上述代码路径、毫秒级模式和同类故障
注入共同锁定根因。最终端到端确认仍以尚未授权的 matched remote rerun 为准。

## 实现的缓解与语义修正

1. `UpstreamPool::acquire` 在复用空闲 socket 前执行无阻塞 `MSG_PEEK`：FIN
   返回 0，reset/其他错误返回非 `EAGAIN`，不应存在的残留字节也视为协议状态
   不安全；上述情况全部先 close + 清空 Beast buffer，再由既有连接流程重连。
2. 不在写出 POST 后自动重试，以免上游其实已执行请求时造成重复推理。
3. admission queue 满时从 503 校正为 Phase 7.1 规定的 429，并保留
   `Retry-After`、`overloaded` 错误码、`reject_overload` 决策和 4xx 计数。
4. 拒绝状态按 `AdmissionKind` 显式分流；queue-full 才返回 429，
   `rejected_unavailable` 继续返回 503/`upstream_unavailable`。新增保护测试防止
   429 默认值误覆盖 unavailable 语义。

## 残余竞态与未实现加固

`MSG_PEEK` 只提供 acquire 时刻的快照。若 FIN 在 probe 完成后、
`async_write` 开始前到达，请求仍可能复用刚刚失效的连接并返回
HTTP 502/`upstream_error`。因此当前改动**只收窄 probe→write 竞态窗口，未关闭
该窗口，也未消除这类传输失败**。

已知但尚未实现的加固是：

- **idle-age eviction**：空闲时间超过上游 keep-alive 安全预算时主动淘汰；
- **max-lifetime eviction**：连接达到绝对生命周期上限后主动轮换。

这两项需要明确的配置、指标和边界测试，留待后续独立改动；本地确定性注入和
尚待执行的 remote matrix 都不能被描述为证明失败类已经消失。

## 改动前失败、改动后通过

测试命令：

```sh
gateway/build/sanitize/tests/gateway_tests \
  '--gtest_filter=GatewayIntegrationTest.ReconnectsBeforeReusingPeerClosedKeepAliveSocket:GatewayIntegrationTest.BoundsQueueAndFastRejectsExcessLoad'
```

- 改动前源码基线：git `527e1bbed0abedc089d59ec96e197cf3d66809fa`，新增测试已存在、
  连接探测与 429 改动尚未应用。结果 0/2：peer-idle-close 的第二请求实际 502（期望
  200），queue-full 实际 503（期望 429）。
- 改动后：2/2 通过；peer-idle-close 测试确认 accepted connection 增加 1，
  queue-full 测试同时确认 429、`Retry-After`、`overloaded`、
  `reject_overload=1`、4xx=1、5xx=0。
- 本次源码与测试补丁 SHA-256：
  `622fbeedceebe154958f635528e11400e2ac7307b782ebfac41647c2a622d6db`。

## 本地验证

```sh
make gateway-test
make test
make phase6-release-write
make reproduce-headline
```

- `make gateway-test`：ASan/UBSan，33/33 通过。
- peer-idle-close、queue-full 429、unavailable 503 三项关键回归用
  `--gtest_repeat=20 --gtest_break_on_failure` 重复 20 轮，共 60/60 次通过；
  该确定性注入仍不覆盖 FIN 落入 probe→write 窗口的情形。
- `make test`：pytest 187/187，Ruff check 与 format check 通过。
- `make phase6-release-write`：通过；`gateway/README.md` 的 release-manifest
  SHA-256 重封为 `b89787916de8dc267a19350093405df9019c312164fb7d56301358b6b47a7c40`。
- `make reproduce-headline`：通过；验证 30 个 source files、16 个 release files、
  4 个 derived outputs；headline SHA-256 保持
  `f409ced2e9d1a08a52e0cb74d955f83c25b697c382a0253a5007bab6c7a909b6`。
- 未运行 remote matrix、远程 vLLM、远程 TSan 或 Phase 7.2。

## Gate 7.1（截至本地会话）

- [x] root cause documented with evidence
- [x] regression tests old-fail/new-pass
- [ ] matched matrix + overload rerun receipts — **按本次范围留给独立远程会话**
- [ ] 429 semantics verified — **本地已验证；远程 overload 回执待补**
- [x] `production blocked` flag lifted or honestly retained — **诚实保留，等待远程回执**
