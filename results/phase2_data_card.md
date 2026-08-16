# Phase 2 teacher-data card

Status: **COMPLETE**.

## Frozen source

- Label-rules version: 3
- Dataset hash: `2f2498c95ea224e48cfc7ee9c705ecef00563c616b0ec14512800706cd8f2573`
- Label-rules SHA-256: `216199a541c360645927fb195d12c0da779e3e4f4743c6dcda189c63aa0e1812`
- TRAIN payload SHA-256: `a753b3de8cbf85aabf71dc06ab960b371c586ff764e8f221c83a889d09676057`
- TEST-IID and TEST-DRIFT were read only by the contamination auditor and were never sent to the teacher.

## Filter funnel

| Stage | Rows | Retained from previous | Retained from selected |
|---|---:|---:|---:|
| selected TRAIN inputs | 6000 | 100.0% | 100.0% |
| teacher response recorded | 6000 | 100.0% | 100.0% |
| schema valid | 6000 | 100.0% | 100.0% |
| verifier accepted | 1664 | 27.7% | 27.7% |
| MinHash unique | 1664 | 100.0% | 27.7% |
| contamination clean | 1450 | 87.1% | 24.2% |

Verifier rejection requires schema validity, scorer-v2 task success, and semantically meaningful tool arguments. MinHash then removes near-duplicate TRAIN narratives. The contamination stage quarantines any survivor with an exact normalized 13-token n-gram found in either frozen TEST split.

## Materialized corpora

| Artifact | Rows | SHA-256 |
|---|---:|---|
| Rule-label SFT | 1450 | `0d54f4ead643d5b7c54247be396780b77e5566837aa8d19fe03e77b6c4a4215e` |
| Distilled SFT | 1450 | `38e2f9f9b990abcf1ea7908fa47e539d8b5864cd9533a3ebcdbcd05d1bbc3fe9` |
| DPO pairs | 1450 | `5b8d32b2e02aeae4f1b60bad6e1ae94ad453f0c18238852e5a09f5839c375522` |

Rule and distilled SFT coverage is identical over 1450 complaint IDs; their ordered complaint-ID SHA-256 is `2e1b4b0c603355d3bd5fd9b60936fc7bc46daad3389a8210b0775ba4310b519e`. DPO chosen responses are surviving high-scoring teacher outputs. Rejected responses are deterministic lower-scoring near misses from the documented perturbation taxonomy.

## Contamination

- Quarantined TRAIN samples: 214
- TEST-IID rows scanned: 104443
- TEST-DRIFT rows scanned: 80000
- Quarantine receipt: `3df6e93574997aee320f48e9a93ad7f103e4121391644df058279d602966ad38`

A zero quarantine count is a clean audit under this exact 13-token policy. A nonzero count is also gate-complete because every hit is excluded and preserved in the committed quarantine receipt.

## Teacher versus frozen rule policy

Denominator: 6000 schema-valid teacher outputs before verifier rejection.

| Field | Matches | Disagreements | Match rate |
|---|---:|---:|---:|
| urgency | 2208 | 3792 | 36.8% |
| ambiguity_flag | 5694 | 306 | 94.9% |
| tool_choice | 2526 | 3474 | 42.1% |
| tool_arguments_structural | 6000 | 0 | 100.0% |

Urgency agreement is 36.8%. This measures teacher compliance with the frozen keyword policy, not human semantic correctness. The Phase 1.2 review established known rule false negatives, so downstream reports must describe urgency ground truth as rule-policy ground truth.

## Cost ledger

- Teacher model: `anthropic/claude-haiku-4.5`
- Provider-receipted cost for the 6,000 unique responses: **$12.667514**
- Replayed 16-item batch charge after one truncated HTTP response: **$0.033176**
- Account-reconciled Phase 2 API cost: **$12.700690**
- Reconciliation: current-key usage `$12.917487` minus the pre-Phase-2 calibration receipt `$0.216797` equals `$12.700690`; equivalently, `$12.667514 + $0.033176 = $12.700690`.
- Transport/account receipt: `c5ff791e5718ab24668942a4740bccf869c2f863379d7a5c19bd0c81a24dfc29`
- Run-specific hard cap: $20.00
- Project teacher-API envelope: $20–$50
- Within run cap: yes
- GPU cost: $0.00; Phase 2 is local plus API only.

The first live attempt stopped after 5,744 durable responses when an HTTP body was truncated. The frozen run resumed from that checkpoint and replayed sequences 5,744–5,759. The unique-response log alone therefore understates account spend by `$0.033176`; `results/phase2_transport_incident.json` preserves the sanitized current-key snapshot, replay receipts, arithmetic, and exception. No API key is committed.

## Reproduction

```bash
make teacher-data SMOKE=1
make teacher-audit SMOKE=1
make teacher-audit
```

The live raw response log is receipt-backed and resumable. Re-auditing also verifies the account reconciliation and makes no network request.
