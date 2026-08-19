SHELL := /bin/sh

SMOKE ?= 0
export SMOKE

LIVE ?= 0
export LIVE

C3_TARGETS := test lint gateway-test gateway-tsan phase1-2 \
	ingest splits calibrate-difficulty \
	teacher-data teacher-audit \
	train-sft train-dpo train-grpo eval export-model \
	serve-bench spec-decode-bench structured-bench bench-report \
	gateway-bench gateway-bench-report sync-up sync-down demo-build reproduce-headline

.PHONY: $(C3_TARGETS) gateway-llama-test ci-lint prepare-r1b prepare-r4-v2 phase3-context-audit phase3-report \
	phase3-preflight phase3-smoke phase4-preflight phase4-smoke phase6-smoke \
	phase6-release-write

test:
	uv run pytest
	uv run ruff check .
	uv run ruff format --check .

lint:
	uv run ruff check .
	uv run ruff format --check .

ci-lint:
	uv run yamllint .github/workflows/ci.yml

gateway-test:
	cmake -S gateway --preset sanitize
	cmake --build gateway/build/sanitize --parallel
	ctest --test-dir gateway/build/sanitize --output-on-failure

gateway-tsan:
	docker build --file gateway/Dockerfile.tsan --tag frontier-forge-gateway-tsan gateway
	docker run --rm --env SMOKE=$(SMOKE) frontier-forge-gateway-tsan

gateway-llama-test:
	cmake -S gateway --preset sanitize
	cmake --build gateway/build/sanitize --parallel
	./gateway/tests/run_llama_cpp_integration.sh

gateway-bench:
	@test "$(shell uname -s)" = "Linux" || { echo "gateway-bench is remote Linux only" >&2; exit 2; }
	@test -n "$(DIRECT_URL)" || { echo "gateway-bench requires DIRECT_URL" >&2; exit 2; }
	@test -n "$(GATEWAY_URL)" || { echo "gateway-bench requires GATEWAY_URL" >&2; exit 2; }
	@./scripts/remote/phase5_gpu_guard.sh
	@.venv-phase4/bin/python gateway/bench/phase5_bench.py \
		--config configs/phase5/gateway_r1b_mtp.yaml \
		--direct-url "$(DIRECT_URL)" --gateway-url "$(GATEWAY_URL)" \
		$(if $(strip $(STAGE)),--stage "$(STAGE)",)

gateway-bench-report:
	@.venv-phase4/bin/python gateway/bench/phase5_report.py \
		--config configs/phase5/gateway_r1b_mtp.yaml

ingest:
	@uv run python -m forge.data.ingest $(if $(filter 1,$(SMOKE)),--smoke,)

ifeq ($(SMOKE),1)
splits: ingest
	@uv run python -m forge.data.splits --smoke

calibrate-difficulty: splits
	@uv run python -m forge.data.calibrate --smoke
else
splits: phase1-2

calibrate-difficulty: phase1-2
endif

phase1-2:
	@uv run python -m forge.teacher.freeze

teacher-data:
	@uv run python -m forge.teacher.generate \
		$(if $(filter 1,$(SMOKE)),--smoke,$(if $(filter 1,$(LIVE)),--live,))

teacher-audit:
	@uv run python -m forge.teacher.audit $(if $(filter 1,$(SMOKE)),--smoke,)

prepare-r1b:
	@uv run python -m forge.train.ablation $(if $(filter 1,$(SMOKE)),--smoke,)

prepare-r4-v2:
	@uv run python -m forge.train.fresh_pool $(if $(filter 1,$(SMOKE)),--smoke,)

phase3-context-audit:
	@uv run --group train python -m forge.train.context_audit

train-sft:
	@uv run --group train python -m forge.train.sft \
		--config $(if $(strip $(CONFIG)),$(CONFIG),configs/r1_sft_rule.yaml) \
		--backend $(if $(strip $(BACKEND)),$(BACKEND),trl) \
		$(if $(strip $(SEED)),--seed $(SEED),) \
		$(if $(filter 1,$(SMOKE)),--smoke,)

train-dpo:
	@uv run --group train python -m forge.train.dpo \
		--config $(if $(strip $(CONFIG)),$(CONFIG),configs/r3_dpo.yaml) \
		--backend $(if $(strip $(BACKEND)),$(BACKEND),trl) \
		$(if $(strip $(SEED)),--seed $(SEED),) \
		$(if $(filter 1,$(SMOKE)),--smoke,)

train-grpo:
	@uv run --group train python -m forge.train.grpo \
		--config $(if $(strip $(CONFIG)),$(CONFIG),configs/r4_grpo.yaml) \
		--backend $(if $(strip $(BACKEND)),$(BACKEND),trl) \
		$(if $(strip $(SEED)),--seed $(SEED),) \
		$(if $(filter 1,$(SMOKE)),--smoke,)

eval:
	@uv run --group train python -m forge.train.evaluate \
		$(if $(strip $(CONFIG)),--config $(CONFIG),--available) \
		--backend $(if $(strip $(BACKEND)),$(BACKEND),trl) \
		$(if $(strip $(SEED)),--seed $(SEED),) \
		$(if $(filter 1,$(SMOKE)),--smoke,)

export-model:
	@uv run --group train python -m forge.train.export \
		--config $(if $(strip $(CONFIG)),$(CONFIG),configs/r4_grpo.yaml) \
		--backend $(if $(strip $(BACKEND)),$(BACKEND),trl) \
		$(if $(strip $(SEED)),--seed $(SEED),) \
		$(if $(filter 1,$(SMOKE)),--smoke,)

phase3-report:
	@uv run python -m forge.train.report $(if $(filter 1,$(SMOKE)),--smoke,)

phase3-preflight:
	@uv run python -m forge.train.preflight \
		$(if $(strip $(CONFIG)),--config $(CONFIG),--all) \
		$(if $(strip $(SEED)),--seed $(SEED),) \
		$(if $(filter 1,$(SMOKE)),--smoke,)

phase3-smoke:
	@test "$(SMOKE)" = "1" || { echo "phase3-smoke requires SMOKE=1" >&2; exit 2; }
	@$(MAKE) prepare-r1b SMOKE=1
	@$(MAKE) train-sft SMOKE=1 CONFIG=configs/r1_sft_rule.yaml
	@$(MAKE) train-sft SMOKE=1 CONFIG=configs/r1b_sft_rule_20k.yaml
	@$(MAKE) train-sft SMOKE=1 CONFIG=configs/r2_sft_distilled.yaml
	@$(MAKE) train-dpo SMOKE=1 CONFIG=configs/r3_dpo.yaml
	@$(MAKE) prepare-r4-v2 SMOKE=1
	@$(MAKE) train-grpo SMOKE=1 CONFIG=configs/r4_grpo.yaml SEED=0
	@$(MAKE) eval SMOKE=1
	@$(MAKE) export-model SMOKE=1 CONFIG=configs/r1b_sft_rule_20k.yaml SEED=0
	@$(MAKE) phase3-report SMOKE=1

phase4-preflight:
ifeq ($(SMOKE),1)
	@uv run python -m forge.bench.preflight
else
	@.venv-phase4/bin/python -m forge.bench.preflight
endif

serve-bench:
ifeq ($(SMOKE),1)
	@uv run python -m forge.bench.smoke \
		--config $(if $(strip $(CONFIG)),$(CONFIG),configs/phase4/serve_r1b_bf16.yaml)
else
	@test -n "$(CONFIG)" || { echo "serve-bench requires CONFIG" >&2; exit 2; }
	@.venv-phase4/bin/python -m forge.bench.runner \
		--config "$(CONFIG)" --base-url "$(if $(strip $(SERVER_URL)),$(SERVER_URL),http://127.0.0.1:8000)"
endif

spec-decode-bench:
ifeq ($(SMOKE),1)
	@uv run python -m forge.bench.smoke \
		--config $(if $(strip $(CONFIG)),$(CONFIG),configs/phase4/spec_r1b_bf16_baseline.yaml)
else
	@test -n "$(CONFIG)" || { echo "spec-decode-bench requires CONFIG" >&2; exit 2; }
	@.venv-phase4/bin/python -m forge.bench.runner \
		--config "$(CONFIG)" --base-url "$(if $(strip $(SERVER_URL)),$(SERVER_URL),http://127.0.0.1:8000)"
endif

structured-bench:
ifeq ($(SMOKE),1)
	@uv run python -m forge.bench.smoke \
		--config $(if $(strip $(CONFIG)),$(CONFIG),configs/phase4/structured_r1b_bf16_xgrammar.yaml)
else
	@test -n "$(CONFIG)" || { echo "structured-bench requires CONFIG" >&2; exit 2; }
	@.venv-phase4/bin/python -m forge.bench.runner \
		--config "$(CONFIG)" --base-url "$(if $(strip $(SERVER_URL)),$(SERVER_URL),http://127.0.0.1:8000)"
endif

bench-report:
ifeq ($(SMOKE),1)
	@uv run python -m forge.bench.report --smoke
else
	@.venv-phase4/bin/python -m forge.bench.report
endif

phase4-smoke:
	@test "$(SMOKE)" = "1" || { echo "phase4-smoke requires SMOKE=1" >&2; exit 2; }
	@$(MAKE) phase4-preflight SMOKE=1
	@$(MAKE) serve-bench SMOKE=1 CONFIG=configs/phase4/serve_r1b_bf16.yaml
	@$(MAKE) serve-bench SMOKE=1 CONFIG=configs/phase4/serve_r1b_gptq_int4.yaml
	@$(MAKE) serve-bench SMOKE=1 CONFIG=configs/phase4/serve_r3eq_bf16.yaml
	@$(MAKE) serve-bench SMOKE=1 CONFIG=configs/phase4/serve_r3eq_gptq_int4.yaml
	@$(MAKE) spec-decode-bench SMOKE=1 CONFIG=configs/phase4/spec_r1b_bf16_baseline.yaml
	@$(MAKE) spec-decode-bench SMOKE=1 CONFIG=configs/phase4/spec_r1b_bf16_mtp.yaml
	@$(MAKE) structured-bench SMOKE=1 CONFIG=configs/phase4/structured_r1b_bf16_xgrammar.yaml
	@$(MAKE) structured-bench SMOKE=1 CONFIG=configs/phase4/structured_r1b_bf16_outlines.yaml
	@$(MAKE) bench-report SMOKE=1

sync-up:
	@./scripts/remote/sync.sh up

sync-down:
	@./scripts/remote/sync.sh down

demo-build:
	@uv run python -m forge.release --demo-build

reproduce-headline:
	@uv run python -m forge.release

phase6-release-write:
	@uv run python -m forge.release --write

phase6-smoke:
	@test "$(SMOKE)" = "1" || { echo "phase6-smoke requires SMOKE=1" >&2; exit 2; }
	@$(MAKE) test SMOKE=1
	@$(MAKE) ingest splits calibrate-difficulty SMOKE=1
	@$(MAKE) teacher-data teacher-audit SMOKE=1
	@FORGE_SMOKE_OUTPUT_ROOT="$(CURDIR)/.tmp-phase6-smoke/$(shell git rev-parse --short=12 HEAD)/phase3" \
		$(MAKE) phase3-smoke SMOKE=1
	@$(MAKE) phase4-smoke SMOKE=1
	@$(MAKE) gateway-test SMOKE=1
	@$(MAKE) reproduce-headline demo-build SMOKE=1
