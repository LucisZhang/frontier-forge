SHELL := /bin/sh

SMOKE ?= 0
export SMOKE

C3_TARGETS := test lint gateway-test gateway-tsan \
	ingest splits calibrate-difficulty \
	teacher-data teacher-audit \
	train-sft train-dpo train-grpo eval export-model \
	serve-bench spec-decode-bench structured-bench bench-report \
	gateway-bench sync-up sync-down demo-build reproduce-headline

STUB_TARGETS := teacher-data teacher-audit \
	train-sft train-dpo train-grpo eval export-model \
	serve-bench spec-decode-bench structured-bench bench-report \
	gateway-bench demo-build reproduce-headline

.PHONY: $(C3_TARGETS) ci-lint

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
else
splits:
	@uv run python -m forge.data.relabel
endif

calibrate-difficulty: splits
	@uv run python -m forge.data.calibrate $(if $(filter 1,$(SMOKE)),--smoke,)

sync-up:
	@./scripts/remote/sync.sh up

sync-down:
	@./scripts/remote/sync.sh down

$(STUB_TARGETS):
	@test "$${SMOKE}" = "$(SMOKE)"
	@printf '[stub] %s\n' "$@"
