# AGENTS.md

## Project Overview
frontier-forge: post-train a 4B base LLM (SFT → distillation → DPO → GRPO) past the
frontier-API cost-quality frontier on a machine-verifiable structured-triage task;
serve it via vLLM behind a custom C++20 LLM-aware gateway. Sister repos:
`~/nlp-eval-lab` (upstream cascade, style reference), `~/batch-recsys-lab`.

Execution phases live in PLAN.md. Locked decisions live in DECISIONS.md — never
re-litigate or silently deviate from them; if a decision proves infeasible, stop
and report, do not improvise a substitute.

## Project Structure
- `src/forge/data` — ingest, labeling, splits (splits are FROZEN after first materialization)
- `src/forge/verify` — rule-based verifier (the foundation; heaviest tests live here)
- `src/forge/teacher` — distillation data factory (API-based, no GPU)
- `src/forge/train` — SFT/DPO/GRPO configs and launchers (TRL reference path, Unsloth fast path)
- `src/forge/serve` — vLLM launch configs, structured-output experiments
- `src/forge/bench` — load generator, metrics collection, report builder
- `src/forge/analysis` — plots, tables, headline computation
- `gateway/` — standalone C++20 CMake project (Boost.Asio, GoogleTest)
- `configs/` — one YAML per experiment, seeds pinned
- `results/runs.jsonl` — APPEND-ONLY; every headline number lives here
- `scripts/remote/` — GPU pod bootstrap + rsync sync
- `demo/` — offline-first static site (built in Phase 6)

## Tech Stack
Python 3.12 + `uv` (locked). PyTorch / Transformers / TRL / Unsloth — all CUDA
paths must also expose `SMOKE=1`. C++20, CMake, vcpkg; clang locally, gcc + TSan
in Linux container. DuckDB for data plumbing (match nlp-eval-lab conventions).

## Development Commands
- Install: `uv sync`
- All tests: `make test` (pytest + ruff)
- Gateway build+test: `make gateway-test` (ASan/UBSan); `make gateway-tsan` (Docker)
- Any pipeline in smoke mode: append `SMOKE=1` (tiny model, ≤100 rows, Mac CPU/MPS)
- Full training/serving: remote GPU only — never attempt CUDA paths locally

## Hardware Rules
- Local = Apple Silicon Mac, 16GB RAM, NO CUDA. vLLM/SGLang/Unsloth/bitsandbytes
  do not run here. Local upstream for gateway dev is llama.cpp server or the mock.
- `SMOKE=1` must pass locally before any remote launch is prepared.
- Remote full runs are launched by the human, not by the agent, unless the human
  explicitly delegates an SSH launch in the current session.

## Hard Rules
- NEVER modify frozen splits, snapshot hashes, or existing runs.jsonl records.
- Every README/report claim must trace to a runs.jsonl record + reproducible command.
- Negative results are recorded and reported, never deleted or hidden.
- Training-time quantization (QLoRA 4-bit) and deployment quantization (AWQ/GPTQ)
  are separate facts — never conflate them in docs or metrics.
- No secrets in the repo; `.env` is gitignored; API keys via env vars only.
- No artifacts >100MB in git (checkpoints go to cloud storage / HF Hub).
- Do not start the next phase; each thread executes exactly the phase it was given.

## PR / Commit Conventions
Conventional commits. One experiment = one config file = one runs.jsonl append.
Every phase ends with: change summary, verification command output, and the
phase's gate checklist from PLAN.md with each item checked or explicitly failed.
