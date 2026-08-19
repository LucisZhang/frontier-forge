# Phase 1.1 calibration remediation report

## Gate result

**ESCALATE.** The local smoke stand-in scored **100.0%** task success on n=13 CAL rows (95% Wilson CI 77.2%–100.0%), **above** the D3 target band of 20.0%–50.0%.

Per D3.1, no difficulty knobs were changed to force the target. The human must review this above-band result before Phase 2.

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
| Urgency match | 100.0% | yes |
| Ambiguity flag match | 100.0% | yes |
| Tool choice match | 100.0% | yes |
| Tool arguments structurally valid | 100.0% | yes |
| Mean scorer-v2 reward | 100.0% | — |

## Secondary metrics (excluded from success and reward)

| Metric | Result |
|---|---:|
| Product match | 100.0% |
| Issue normalized match | 100.0% |
| Company normalized match | 100.0% |
| Tool argument semantic validity | 100.0% |
| Abstention correctness | 100.0% |

## Delta from Phase 1

| Contract | Phase 1 | Phase 1.1 |
|---|---|---|
| Rows | 25 | 13 |
| Task success | 0.0% | 100.0% |
| Visible evidence | complaint ID + narrative | narrative + source metadata |
| Success checks | hard-AND over 8 checks | decision fields + structural args |
| Tool free text | normalized template equality | structural + non-verbatim semantics |
| Label rules | v1 phrase trigger at any length | v2 phrase trigger capped at 200 chars |

## Reproducibility and receipts

- CAL artifact: `data/smoke/splits/cal.parquet`
- CAL payload SHA-256: `5a3d4f74010e192151c1b86eafb592fb2ac7a4abf59ab537f9c594b9b4a545fe`
- Scorer version: `2`
- Input contract version: `2`
- Visible input fields: `complaint_id, narrative, source_product, source_issue, source_company`
- Append-only run record: `results/runs.jsonl` entry `phase1_1_api_calibration_v2_s20260815`
- Offline report command: `make calibrate-difficulty`
- Network calls: 0 (smoke plumbing only; not a gate result)

**HUMAN REVIEW REQUIRED before Phase 2 starts.**
