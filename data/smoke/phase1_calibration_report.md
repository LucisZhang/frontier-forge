# Phase 1 difficulty calibration report

## Decision status

**HUMAN DECISION REQUIRED.** No final difficulty setting or D3 task fallback is selected here.

The SMOKE stand-in scored **0.0%** task success (95% Wilson CI 0.0%–22.8%), which is **below** the D3 target band of 20.0%–50.0%.

This is plumbing evidence from `deterministic-rule-blind-stand-in-v1`, not a Qwen base-model result. It makes no model or API calls.

## Evaluated difficulty knobs

- Candidate: `c1_full_ticket_v1` (proposed for human review)
- Schema fields: 6
- Product classes: 9
- Tools: 5 plus 2 distractor slots
- Ambiguous-sample ratio: `natural_rule_rate`
- Bilingual instructions: `false`
- Issue scoring: `normalized_exact`

Alternatives retained for the owner: `c0_reduced_ticket` (easier) and `c2_distractor_heavy` (harder). Changing to either requires an explicit human decision before any new split materialization; frozen artifacts are never edited in place.

## Stand-in metrics

| Metric | Result |
|---|---:|
| Samples | 13 |
| Task success | 0.0% |
| Schema valid | 100.0% |
| Product match | 15.4% |
| Issue match | 0.0% |
| Company match | 0.0% |
| Urgency match | 100.0% |
| Ambiguity match | 0.0% |
| Tool choice | 0.0% |
| Tool arguments | 0.0% |
| Abstention correctness | 0.0% |
| Mean verifier reward | 31.5% |

## Reproducibility

- CAL artifact: `/Users/hsiangkuochang/frontier-forge/data/smoke/splits/cal.parquet`
- CAL SHA-256: `5a3d4f74010e192151c1b86eafb592fb2ac7a4abf59ab537f9c594b9b4a545fe`
- Command: `make calibrate-difficulty` (or `SMOKE=1 make calibrate-difficulty`)
- Network calls: 0 for this stand-in run
