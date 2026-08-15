# Phase 1 teacher spot-label audit

## Decision status

**HUMAN REVIEW REQUIRED.** The teacher audit is evidence about the rule labels; it
does not authorize editing the frozen splits or selecting final difficulty knobs.

Twenty deterministic ambiguous CAL rows were audited twice with
`anthropic/claude-haiku-4.5` through OpenRouter. No TRAIN, TEST-IID, or TEST-DRIFT
row was sent.

## Batch results

| Batch | Prompt | Calls | Schema-valid | Exact rule agreement | Reported API USD |
|---|---|---:|---:|---:|---:|
| v1 negative result | `phase1_spot_label_v1.txt` | 20 | 0 | 0 | $0.040826 |
| v2 explicit C1 contract | `phase1_spot_label_v2.txt` | 20 | 20 | 0 | $0.035586 |
| **Total** |  | **40** | **20** | **0** | **$0.076412** |

The v1 provider ignored the structured-response constraint and returned fenced,
non-C1 objects. That failed batch and its costs are retained, not rewritten or
discarded. The v2 prompt made every allowed field, enum, and tool-argument shape
visible in the message; all 20 outputs then satisfied C1.

## v2 agreement breakdown

| Check | Matches / 20 |
|---|---:|
| Product | 19 |
| Issue | 5 |
| Company | 19 |
| Urgency | 7 |
| Ambiguity flag | 1 |
| Tool choice | 1 |
| Tool arguments | 0 |
| Full task success | 0 |

The rule table sent all 20 sampled rows to `request_more_info`; the teacher sent
only one there and instead chose 11 `route_to_company`, 6
`escalate_to_regulator`, 1 `start_refund_workflow`, and 1 `close_no_action`.
Mean verifier agreement reward was 0.465.

This is strong disagreement evidence around the current short/ambiguous rule, not
permission to prefer the teacher automatically. The owner should review these 20
receipts alongside the 200-row rule-label audit before accepting the candidate or
authorizing a new versioned materialization. Existing frozen artifacts remain
unchanged.

## Receipts and budget proof

- v1 ledger: `results/phase1_teacher_spot_label_ledger.json`
- v1 receipts SHA-256:
  `a6cfbef0bb42d86ccd3b291124541e87b677cf7f0f8ed73b590a8c5ab88bedb7`
- v2 ledger: `results/phase1_teacher_spot_label_ledger_v2.json`
- v2 receipts SHA-256:
  `938cbdb08387230dcfb154d822c405e8ab1914e092cac615e5eac35365857df7`
- Budget ceiling: $10.00; combined reported spend: $0.076412; missing cost
  receipts: 0.

