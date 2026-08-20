# Phase 7.1 evidence handling notes

- `a10_bootstrap-mirror-sdk-attempt.log` retains the failed Hugging Face mirror SDK download attempt and its three timeout contexts. The temporary CDN signed query strings were replaced with `[signed-query-redacted]` before commit; paths, timestamps, retry progress, exception types, and outcomes remain intact.
- Synced `.log` files had trailing spaces and CRLF line endings normalized for repository hygiene; no semantic log content was removed.
- Benchmark receipts under `raw/`, request schedules under `requests/`, the GPU ledger, host/toolchain/model receipts, and service-shutdown receipts are synchronized byte-for-byte from the delegated A10 VM.
- Gitleaks flags one JWT-shaped value in `a10_mirror_prefill_pip_report.json`; inspection locates it in the public PyJWT package-description example embedded by pip. It is not a project credential and is retained so the pip report continues to match `a10_mirror_prefill_receipt.json`'s recorded SHA-256.
- `README.md` was intentionally not updated because Gate 7.1 failed its locked per-overload-cell 429 acceptance check.
