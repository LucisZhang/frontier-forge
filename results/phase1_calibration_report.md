# Phase 1 difficulty calibration report

## Decision status

**HUMAN DECISION REQUIRED.** No final difficulty setting or D3 task fallback is selected here.

The receipt-backed API stand-in scored **0.0%** task success on 25 frozen CAL rows (95% Wilson CI 0.0%–13.3%), which is **below** the D3 target band of 20.0%–50.0%.

This is zero-shot evidence from `anthropic/claude-haiku-4.5`, with only complaint ID and narrative visible to the model. It is a small API stand-in, not the Qwen base-model result. The first valid full calibration remains the human-launched Qwen base run in Phase 3.

## Evaluated difficulty knobs

- Candidate: `c1_full_ticket_v1` (proposed for human review)
- Schema fields: 6
- Product classes: 9
- Tools: 5 plus 2 distractor slots
- Ambiguous-sample ratio: `natural_rule_rate`
- Bilingual instructions: `false`
- Issue scoring: `normalized_exact`

Alternatives retained for the owner: `c0_reduced_ticket` (easier) and `c2_distractor_heavy` (harder). Changing to either requires an explicit human decision before any new split materialization; frozen artifacts are never edited in place.

## API stand-in metrics

| Metric | Result |
|---|---:|
| Samples | 25 |
| Task success | 0.0% |
| Schema valid | 100.0% |
| Product match | 80.0% |
| Issue match | 0.0% |
| Company match | 0.0% |
| Urgency match | 44.0% |
| Ambiguity match | 52.0% |
| Tool choice | 16.0% |
| Tool arguments | 0.0% |
| Abstention correctness | 52.0% |
| Mean verifier reward | 44.4% |

## Deterministic plumbing baseline

`deterministic-rule-blind-stand-in-v1` made zero network calls and scored 0.0% task success on 200 CAL rows. It remains a pipeline check only.

## Reproducibility

- CAL artifact: `/Users/hsiangkuochang/frontier-forge/data/splits/cal.parquet`
- CAL SHA-256: `2bc6e3886319d78afdcd35c45d9a1e81d0f06209b477dd04da7744a6c7ad8074`
- Command: `make calibrate-difficulty` (or `SMOKE=1 make calibrate-difficulty`)
- API stand-in command: `python -m forge.data.api_calibrate --live`; subsequent report builds consume the frozen receipts offline
- API calls: 25
- API selection: smallest `blake2b('20260815:<complaint_id>')`, cap 25
- Reported API cost: $0.044239
- API receipts SHA-256: `c5f5636fc5eb128a8e8174a8d9e328be6946cc27dfa26ecb93e0aec32ac987c7`
- Deterministic selection: same rank, cap 200
