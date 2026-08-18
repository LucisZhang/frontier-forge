"""Run one Phase 4 experiment end-to-end against the local smoke server."""

from __future__ import annotations

import argparse

from .config import load_phase4_config
from .runner import run
from .smoke_server import SmokeServer
from .workload import build_workload, load_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    build_workload(args.config, smoke=True)
    workload = load_workload(config, smoke=True)
    speculative = bool(config.get("speculative", {}).get("enabled", False))
    with SmokeServer(
        workload,
        model=str(config["model"]["served_name"]),
        speculative=speculative,
    ) as server:
        run(args.config, base_url=server.base_url, smoke=True)


if __name__ == "__main__":
    main()
