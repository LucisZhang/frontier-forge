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
	gateway-bench sync-up sync-down demo-build reproduce-headline

STUB_TARGETS := serve-bench spec-decode-bench structured-bench bench-report \
	gateway-bench demo-build reproduce-headline

.PHONY: $(C3_TARGETS) ci-lint prepare-r1b prepare-r4-v2 phase3-context-audit phase3-report \
	phase3-preflight phase3-smoke

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

sync-up:
	@./scripts/remote/sync.sh up

sync-down:
	@./scripts/remote/sync.sh down

$(STUB_TARGETS):
	@test "$${SMOKE}" = "$(SMOKE)"
	@printf '[stub] %s\n' "$@"
