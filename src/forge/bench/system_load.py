"""Host CPU co-tenancy sampling for latency-sensitive Phase 4 sweeps."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class CpuTimes:
    idle: int
    total: int


def _read_linux_cpu_times(path: Path = Path("/proc/stat")) -> CpuTimes | None:
    try:
        fields = path.read_text().splitlines()[0].split()
        if not fields or fields[0] != "cpu":
            return None
        values = [int(value) for value in fields[1:]]
    except (FileNotFoundError, IndexError, ValueError):
        return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return CpuTimes(idle=idle, total=sum(values))


def _cpu_utilization(previous: CpuTimes | None, current: CpuTimes | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current.total - previous.total
    idle_delta = current.idle - previous.idle
    if total_delta <= 0:
        return None
    return min(1.0, max(0.0, 1.0 - idle_delta / total_delta))


class SystemLoadSampler:
    """Sample host load and aggregate CPU utilization without extra dependencies."""

    def __init__(self, *, enabled: bool, interval_s: float = 0.2) -> None:
        self.enabled = enabled
        self.interval_s = interval_s
        self.logical_cpu_count = os.cpu_count() or 1
        self.load_threshold = self.logical_cpu_count / 2
        self.samples: list[dict[str, float | None]] = []
        self._stop = asyncio.Event()
        self._previous_cpu: CpuTimes | None = None

    def sample_once(self) -> None:
        if not self.enabled:
            return
        try:
            load1, load5, load15 = os.getloadavg()
        except OSError:
            load1 = load5 = load15 = 0.0
        current = _read_linux_cpu_times()
        utilization = _cpu_utilization(self._previous_cpu, current)
        self._previous_cpu = current
        self.samples.append(
            {
                "load1": float(load1),
                "load5": float(load5),
                "load15": float(load15),
                "cpu_utilization": utilization,
            }
        )

    async def run(self) -> None:
        if not self.enabled:
            return
        self.sample_once()
        while not self._stop.is_set():
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            if not self._stop.is_set():
                self.sample_once()
        self.sample_once()

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict[str, Any]:
        loads1 = [float(item["load1"] or 0.0) for item in self.samples]
        loads5 = [float(item["load5"] or 0.0) for item in self.samples]
        loads15 = [float(item["load15"] or 0.0) for item in self.samples]
        cpu = [
            float(item["cpu_utilization"])
            for item in self.samples
            if item["cpu_utilization"] is not None
        ]
        maximum = max(loads1) if loads1 else None
        return {
            "measurement_side": "server_host_getloadavg_and_proc_stat",
            "logical_cpu_count": self.logical_cpu_count,
            "load1_contamination_threshold": self.load_threshold,
            "samples": len(self.samples),
            "load1_mean": fmean(loads1) if loads1 else None,
            "load1_max": maximum,
            "load5_mean": fmean(loads5) if loads5 else None,
            "load5_max": max(loads5) if loads5 else None,
            "load15_mean": fmean(loads15) if loads15 else None,
            "load15_max": max(loads15) if loads15 else None,
            "cpu_utilization_mean": fmean(cpu) if cpu else None,
            "cpu_utilization_max": max(cpu) if cpu else None,
            "contaminated": maximum is not None and maximum > self.load_threshold,
            "policy": "rerun sweep when any sampled 1-minute load exceeds half core count",
        }


def metrics_contaminated(metrics: dict[str, Any], *, experiment: str) -> bool:
    """Return whether a completed measurement attempt violates the co-tenancy gate."""

    if experiment in {"serve", "spec_decode"}:
        return any(
            bool(point.get("co_tenancy", {}).get("contaminated"))
            for point in metrics.get("points", [])
        )
    return bool(metrics.get("co_tenancy", {}).get("contaminated"))
