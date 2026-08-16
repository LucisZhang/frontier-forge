# D3.1 label freeze — Phase 2 source lineage

Status: **FROZEN on 2026-08-16 before Phase 2 data generation**.

The human-reviewed label-rules-v3 lineage is accepted for Phase 2 with known false negatives documented in `results/phase1_2_label_audit.md`. Freeze means reproducible immutability, not a claim that the labels are error-free.

## Frozen identity

- Dataset version: 3
- Dataset hash: `2f2498c95ea224e48cfc7ee9c705ecef00563c616b0ec14512800706cd8f2573`
- Phase 1.2 manifest SHA-256: `dc8299726faf20a8d300f510a69f05566f150a319c6dcf43c5f9241cc3c28407`
- Label-rules version: 3
- Label-rules SHA-256: `216199a541c360645927fb195d12c0da779e3e4f4743c6dcda189c63aa0e1812`
- Input contract version: 2
- Scorer version: 2

## Frozen split payloads and memberships

| Split | Rows | Payload SHA-256 | Membership SHA-256 |
|---|---:|---|---|
| TRAIN | 300,000 | `a753b3de8cbf85aabf71dc06ab960b371c586ff764e8f221c83a889d09676057` | `7f63a4afb0b94b5adb7e30d761941e8d76e65f345dc16de85493a1e32134b483` |
| CAL | 86,972 | `e83120a829843d19a9175e21c93315520f287fc5d2955aefde4b1b2f38838dbc` | `f7290f5a4f97a2a020b7ed53711227010764cd48700280fc5742b5d95635e765` |
| TEST-IID | 104,443 | `c196fd9ac620640dbb9fa545a46830300a4699fcda5d2606b018f5417765d6e6` | `049dce9c635a4280eea7b19948807ed20975d134b9e36655645320afad7fbb21` |
| TEST-DRIFT | 80,000 | `6ea9f6210898fb9a72c1bc66146c1e033af00cd5c125607e4e953aff7333dd80` | `5b3d77cee9e3852747f44ee36d8a91fa421b3f24bf28d3fba418aed75fd95946` |

## Boundary

Every Phase 2 generation, corpus, DPO pair, filter audit, and run record must pin the dataset and rule hashes above. TEST-IID and TEST-DRIFT are audit-only inputs for contamination checking and must never be sent to the teacher. No future work may edit the frozen v3 label payloads, their hashes, or prior `results/runs.jsonl` entries.

Reproduction anchor before Phase 2: `make phase1-2` must remain a frozen no-op for this lineage.
