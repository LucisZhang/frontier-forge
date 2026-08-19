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
| selected TRAIN inputs | 10 | 100.0% | 100.0% |
| teacher response recorded | 10 | 100.0% | 100.0% |
| schema valid | 8 | 80.0% | 80.0% |
| verifier accepted | 6 | 75.0% | 60.0% |
| MinHash unique | 6 | 100.0% | 60.0% |
| contamination clean | 6 | 100.0% | 60.0% |

Verifier rejection requires schema validity, scorer-v2 task success, and semantically meaningful tool arguments. MinHash then removes near-duplicate TRAIN narratives. The contamination stage quarantines any survivor with an exact normalized 13-token n-gram found in either frozen TEST split.

## Materialized corpora

| Artifact | Rows | SHA-256 |
|---|---:|---|
| Rule-label SFT | 6 | `cf1c0fb7bbd9ff0ce47efa6a83e05aeeadc594ed1360a5b80c9f45374112dde4` |
| Distilled SFT | 6 | `5c9f0d063d955daa0573c2642698cc52ebd07c099cb2a2f9994da0a0a4fd4d6a` |
| DPO pairs | 6 | `8704197eacab8a3cf32c4c4616af5cfbeb3a225a622a221cbd8d2f1df4408652` |

Rule and distilled SFT coverage is identical over 6 complaint IDs; their ordered complaint-ID SHA-256 is `e5658adc7dc03c2da39b0edc24c385562d901b1d494adb084f4c54ffc542e891`. DPO chosen responses are surviving high-scoring teacher outputs. Rejected responses are deterministic lower-scoring near misses from the documented perturbation taxonomy.

## Contamination

- Quarantined TRAIN samples: 0
- TEST-IID rows scanned: 104443
- TEST-DRIFT rows scanned: 80000
- Quarantine receipt: `662129f490931d6448670299d978266c1a9f4f2afced732c1efa134b86618313`

A zero quarantine count is a clean audit under this exact 13-token policy. A nonzero count is also gate-complete because every hit is excluded and preserved in the committed quarantine receipt.

## Teacher versus frozen rule policy

Denominator: 8 schema-valid teacher outputs before verifier rejection.

| Field | Matches | Disagreements | Match rate |
|---|---:|---:|---:|
| urgency | 6 | 2 | 75.0% |
| ambiguity_flag | 8 | 0 | 100.0% |
| tool_choice | 8 | 0 | 100.0% |
| tool_arguments_structural | 8 | 0 | 100.0% |

Urgency agreement is 75.0%. This measures teacher compliance with the frozen keyword policy, not human semantic correctness. The Phase 1.2 review established known rule false negatives, so downstream reports must describe urgency ground truth as rule-policy ground truth.

## Cost ledger

- Teacher model: `mock/phase2-deterministic-v1`
- Provider-reported Phase 2 API cost: **$0.000000**
- Run-specific hard cap: $0.00
- Project teacher-API envelope: $20–$50
- Within run cap: yes
- GPU cost: $0.00; Phase 2 is local plus API only.

## Reproduction

```bash
make teacher-data SMOKE=1
make teacher-audit SMOKE=1
make teacher-audit
```

The live raw response log is receipt-backed and resumable. Re-auditing makes no network request.
