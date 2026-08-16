# Phase 1.2 calibration receipt re-score

## Gate result

**UNAFFECTED / NO DISCRIMINATIVE POWER.** The same 100 recorded model outputs score **21.0%** task success against label rules v3 (95% Wilson CI 14.2%–30.0%), versus **21.0%** against v2 (delta +0.0 percentage points). All 100 calibration gold labels were unchanged under v3, so this re-score cannot corroborate, validate, or distinguish the v3 rule change; it only proves that the recorded calibration slice was unaffected.

No API request was made: this is an offline re-score of immutable Phase 1.1 receipts, not a new calibration run or a Phase-3 R0 result. The unchanged point estimate remains numerically inside the pre-existing 20–50% calibration band, but it is not evidence for the quality of v3 because 0/100 rows exercised the changed labels.

## Decision-check delta

| Check | v2 receipts | v3 re-score | Delta |
|---|---:|---:|---:|
| Task success | 21.0% | 21.0% | +0.0 pp |
| Schema valid | 100.0% | 100.0% | +0.0 pp |
| Urgency match | 26.0% | 26.0% | +0.0 pp |
| Ambiguity flag match | 90.0% | 90.0% | +0.0 pp |
| Tool choice match | 31.0% | 31.0% | +0.0 pp |
| Tool arguments structurally valid | 100.0% | 100.0% | +0.0 pp |
| Mean scorer-v2 reward | 69.4% | 69.4% | +0.0 pp |

## Gold-label delta on the 100 receipt IDs

- Gold labels changed from v2 to v3: 0/100.
- Gold tool transitions: none.
- Task-success transitions: false -> false=79, true -> true=21.

## Frozen-membership proof

| Split | Rows | Phase 1 SHA-256 | v2 SHA-256 | v3 SHA-256 | Match |
|---|---:|---|---|---|---|
| train | 300000 | `7f63a4afb0b94b5adb7e30d761941e8d76e65f345dc16de85493a1e32134b483` | `7f63a4afb0b94b5adb7e30d761941e8d76e65f345dc16de85493a1e32134b483` | `7f63a4afb0b94b5adb7e30d761941e8d76e65f345dc16de85493a1e32134b483` | yes |
| cal | 86972 | `f7290f5a4f97a2a020b7ed53711227010764cd48700280fc5742b5d95635e765` | `f7290f5a4f97a2a020b7ed53711227010764cd48700280fc5742b5d95635e765` | `f7290f5a4f97a2a020b7ed53711227010764cd48700280fc5742b5d95635e765` | yes |
| test_iid | 104443 | `049dce9c635a4280eea7b19948807ed20975d134b9e36655645320afad7fbb21` | `049dce9c635a4280eea7b19948807ed20975d134b9e36655645320afad7fbb21` | `049dce9c635a4280eea7b19948807ed20975d134b9e36655645320afad7fbb21` | yes |
| test_drift | 80000 | `5b3d77cee9e3852747f44ee36d8a91fa421b3f24bf28d3fba418aed75fd95946` | `5b3d77cee9e3852747f44ee36d8a91fa421b3f24bf28d3fba418aed75fd95946` | `5b3d77cee9e3852747f44ee36d8a91fa421b3f24bf28d3fba418aed75fd95946` | yes |

- Version-2 dataset hash: `de34e40bee09fb452b12aba2659533fc491be94bed6361c639be343221e0393d`
- Version-3 dataset hash: `2f2498c95ea224e48cfc7ee9c705ecef00563c616b0ec14512800706cd8f2573`
- Label-rules-v3 SHA-256: `216199a541c360645927fb195d12c0da779e3e4f4743c6dcda189c63aa0e1812`
- Changed-row audit: `/Users/hsiangkuochang/frontier-forge/results/phase1_2_label_audit.md`
- Changed labels by split: train=4224, cal=374, test_iid=558, test_drift=606

## Receipt integrity and reproducibility

- Existing receipt file: `/Users/hsiangkuochang/frontier-forge/results/phase1_1_api_calibration_receipts.jsonl`
- Existing receipt SHA-256: `00e37c6dbcfbff9e0175be1a59dbe3225abf8742deda8e0fad986072999bb10d`
- Existing provider calls recorded: 100
- New network calls: 0
- Scorer version: `2`
- Model identity preserved from receipts: `anthropic/claude-haiku-4.5`
- Append-only run record: `results/runs.jsonl` entry `phase1_2_api_calibration_v3_offline_rescore_s20260815`
- Reproduction command: `make phase1-2`

## Known limitations intentionally left open

- Negation-blind keyword matching remains (the delegated review estimated about 18 affected rows containing negated `identity theft`).
- The single-action taxonomy remains: escalation outranks refund for dual-remedy narratives.

**HUMAN REVIEW COMPLETE.** The final label review is recorded in `results/phase1_2_label_audit.md`; label rules v3 are frozen under D3.1 before Phase 2 data generation.
