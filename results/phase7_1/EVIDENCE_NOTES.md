# Phase 7.1 evidence handling notes

- `a10_bootstrap-mirror-sdk-attempt.log` retains the failed Hugging Face mirror SDK download attempt and its three timeout contexts. The temporary CDN signed query strings were replaced with `[signed-query-redacted]` before commit; paths, timestamps, retry progress, exception types, and outcomes remain intact.
- Synced `.log` files had trailing spaces and CRLF line endings normalized for repository hygiene; no semantic log content was removed.
- Benchmark receipts under `raw/`, request schedules under `requests/`, the GPU ledger, host/toolchain/model receipts, and service-shutdown receipts are synchronized byte-for-byte from the delegated A10 VM.
- Gitleaks flags one JWT-shaped value in `a10_mirror_prefill_pip_report.json`; inspection locates it in the public PyJWT package-description example embedded by pip. It is not a project credential and is retained so the pip report continues to match `a10_mirror_prefill_receipt.json`'s recorded SHA-256.
- `README.md` was intentionally not updated because Gate 7.1 failed its locked per-overload-cell 429 acceptance check.

## Sustained-load amendment (2026-08-21)

- The prior sentence remains the disposition of the finite-burst run. The human-approved Gate 7.1 amendment replaced only the miscalibrated “one 429 in every 60-request burst” proxy with duration-based arrivals of at least 120 seconds at 2×/3×/5×. The sustained receipt passed and the README was updated by the guarded report writer.
- The original remote checkout contained untracked copies of the already-synchronized finite evidence. Git correctly refused a checkout that would overwrite them, so the sustained run used the isolated worktree `/mnt/frontier-forge/worktrees/phase7-2-a10`; the original checkout and evidence were not moved or deleted.
- The first artifact verification attempt correctly matched the model tree SHA-256 but compared a symlink-resolved physical path against the manifest's logical repository path. The corrected verifier records both paths and still fails closed on the model tree hash and export-manifest hash.
- The bare-vLLM 5× cell triggered a vLLM 0.17.0 GDN+native-MTP EngineCore assertion after 490×200 and 36×500; 651 later requests recorded transport errors. There was no host OOM, NVIDIA Xid, or co-tenancy contamination. The full 49,552,446-byte log SHA-256 is `8b8338f7a19b53a8847d7b505cae2897a7bfc9ecccf39c02d7e0d0554fcee24a`; its lossless gzip copy is committed at 320,357 bytes with SHA-256 `8ed49ff561842d6140b281125a35667ca69f503df1c7079e749069cd1f1f15c3`.
- Both request stages were generated at git `aacacdd89de5a74deb005ea86ba16dff8f74c1cb`. The append-only finalizer at git `a0202a11d000a75b1e0eac0d3e660de875544f51` fixed only the summary parser's handling of the non-numeric `transport_error` status bucket; it revalidated both raw JSON files, both JSONL hashes, config hash, artifact hash, measurement SHA, and paired schedules without rerunning requests. Both SHAs are explicit in the final receipt.
- No security-group or public-listener change was performed. vLLM and the gateway listened only on `127.0.0.1:8000` and `127.0.0.1:9000`.
