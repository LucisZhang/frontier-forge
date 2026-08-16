# Phase 1.1 calibration remediation report

## Gate result

**PASS.** The receipt-backed API stand-in scored **21.0%** task success on n=100 CAL rows (95% Wilson CI 14.2%–30.0%), **inside** the D3 target band of 20.0%–50.0%.

No calibration-band escalation is required.

This remains an API stand-in, not the Phase-3 Qwen base-model R0 result.

## D3.1 contracts implemented

- Input contract v2: narrative plus source product, issue, and company metadata (and complaint ID for tool arguments).
- Scorer v2: task success requires schema validity plus urgency, ambiguity flag, tool choice, and structural argument validity.
- Product/issue/company normalization and non-verbatim tool-text semantics are secondary diagnostics excluded from task success and reward.
- Fair prompt v2 discloses the urgency policy, ambiguity definition, product mapping, tool registry semantics, priority order, and argument schemas.
- Label rules v2 cap phrase-only ambiguity triggers at 200 narrative characters; long narratives containing phrases such as `not sure` are not made ambiguous by that phrase alone.

## Decision-check breakdown

| Check | Result | Included in task success |
|---|---:|---|
| Schema valid | 100.0% | yes |
| Urgency match | 26.0% | yes |
| Ambiguity flag match | 90.0% | yes |
| Tool choice match | 31.0% | yes |
| Tool arguments structurally valid | 100.0% | yes |
| Mean scorer-v2 reward | 69.4% | — |

## Secondary metrics (excluded from success and reward)

| Metric | Result |
|---|---:|
| Product match | 100.0% |
| Issue normalized match | 81.0% |
| Company normalized match | 100.0% |
| Tool argument semantic validity | 100.0% |
| Abstention correctness | 90.0% |

## Delta from Phase 1

| Contract | Phase 1 | Phase 1.1 |
|---|---|---|
| Rows | 25 | 100 |
| Task success | 0.0% | 21.0% |
| Visible evidence | complaint ID + narrative | narrative + source metadata |
| Success checks | hard-AND over 8 checks | decision fields + structural args |
| Tool free text | normalized template equality | structural + non-verbatim semantics |
| Label rules | v1 phrase trigger at any length | v2 phrase trigger capped at 200 chars |

## Frozen-membership proof

| Split | Rows | Phase 1 membership SHA-256 | v2 membership SHA-256 | Match |
|---|---:|---|---|---|
| cal | 86972 | `f7290f5a4f97a2a020b7ed53711227010764cd48700280fc5742b5d95635e765` | `f7290f5a4f97a2a020b7ed53711227010764cd48700280fc5742b5d95635e765` | yes |
| test_drift | 80000 | `5b3d77cee9e3852747f44ee36d8a91fa421b3f24bf28d3fba418aed75fd95946` | `5b3d77cee9e3852747f44ee36d8a91fa421b3f24bf28d3fba418aed75fd95946` | yes |
| test_iid | 104443 | `049dce9c635a4280eea7b19948807ed20975d134b9e36655645320afad7fbb21` | `049dce9c635a4280eea7b19948807ed20975d134b9e36655645320afad7fbb21` | yes |
| train | 300000 | `7f63a4afb0b94b5adb7e30d761941e8d76e65f345dc16de85493a1e32134b483` | `7f63a4afb0b94b5adb7e30d761941e8d76e65f345dc16de85493a1e32134b483` | yes |

- Version-2 dataset hash: `de34e40bee09fb452b12aba2659533fc491be94bed6361c639be343221e0393d`
- Changed-row audit: `/Users/hsiangkuochang/frontier-forge/results/phase1_1_label_audit.md`
- Changed labels by split: cal=1181, test_drift=617, test_iid=1224, train=4590

## Reproducibility and receipts

- CAL artifact: `/Users/hsiangkuochang/frontier-forge/data/phase1_1/splits/cal.parquet`
- CAL payload SHA-256: `c70ff32a7e412f0e0a3974c14d0b3a81afeb14568b155c77eca131c196d9ef83`
- Scorer version: `2`
- Input contract version: `2`
- Visible input fields: `complaint_id, narrative, source_product, source_issue, source_company`
- Append-only run record: `results/runs.jsonl` entry `phase1_1_api_calibration_v2_s20260815`
- Offline report command: `make calibrate-difficulty`
- Live command: `python -m forge.data.api_calibrate --live`
- Model: `anthropic/claude-haiku-4.5`
- Calls: 100
- Reported API cost: $0.216797
- Budget ceiling: $15.00
- Prompt SHA-256: `433ab8819aa5e3f5898131b5e459d20757023cdc9aa84790c97355e3fd31c5fd`
- Receipts SHA-256: `00e37c6dbcfbff9e0175be1a59dbe3225abf8742deda8e0fad986072999bb10d`

**HUMAN REVIEW REQUIRED before Phase 2 starts.**
