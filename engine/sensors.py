"""Machine sensors: what the hardware can spare right now.

GPU numbers come from NVML (nvidia-ml-py), everything else from psutil.
Processes that belong to the learner itself (this process, its children and any
registered worker pids) are never counted as "someone else using the machine".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil

try:  # NVML is optional so the engine (and its tests) run on machines without NVIDIA.
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None

MB = 1024 * 1024


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str
    used_mb: float | None  # None when the driver does not report it


@dataclass(frozen=True)
class Snapshot:
    taken_at: float
    gpu_present: bool
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    gpu_util_pct: float = 0.0
    gpu_temp_c: float = 0.0
    others_on_gpu: tuple[GpuProcess, ...] = ()
    ram_available_mb: float = 0.0
    cpu_others_pct: float = 0.0  # system CPU use minus the learner's own processes
    on_ac: bool = True
    battery_pct: float | None = None

    @property
    def others_vram_mb(self) -> float:
        return sum(p.used_mb or 0.0 for p in self.others_on_gpu)

    @property
    def vram_for_us_mb(self) -> float:
        """GPU memory the learner could use if it wanted: total minus what others hold."""
        return max(0.0, self.vram_total_mb - self.others_vram_mb)


@dataclass
class SystemSensors:
    """Reads the real machine. Construct once and call read() periodically."""

    gpu_index: int = 0
    ignore_process_names: tuple[str, ...] = ()  # e.g. a compositor that always sits on the GPU
    extra_own_pids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except pynvml.NVMLError:
                self._handle = None
        self._me = psutil.Process()
        self._cpu_procs: dict[int, psutil.Process] = {}
        psutil.cpu_percent(None)  # prime the system counter

    def register_own_pid(self, pid: int) -> None:
        self.extra_own_pids.add(pid)

    def unregister_own_pid(self, pid: int) -> None:
        self.extra_own_pids.discard(pid)

    def own_pids(self) -> set[int]:
        pids = {self._me.pid} | set(self.extra_own_pids)
        try:
            pids |= {c.pid for c in self._me.children(recursive=True)}
        except psutil.Error:
            pass
        return pids

    def _own_cpu_pct(self, own: set[int]) -> float:
        """CPU used by the learner's own processes, as a share of the whole machine."""
        total = 0.0
        for pid in own:
            proc = self._cpu_procs.get(pid)
            try:
                if proc is None:
                    proc = psutil.Process(pid)
                    proc.cpu_percent(None)  # first call only primes the counter
                    self._cpu_procs[pid] = proc
                    continue
                total += proc.cpu_percent(None)
            except psutil.Error:
                self._cpu_procs.pop(pid, None)
        for pid in list(self._cpu_procs):
            if pid not in own:
                del self._cpu_procs[pid]
        return total / (psutil.cpu_count() or 1)

    def _gpu(self, own: set[int]) -> dict:
        if self._handle is None:
            return {"gpu_present": False}
        h = self._handle
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        try:
            util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        except pynvml.NVMLError:
            util = 0.0
        try:
            temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        except pynvml.NVMLError:
            temp = 0.0
        seen: dict[int, GpuProcess] = {}
        for getter in (
            pynvml.nvmlDeviceGetComputeRunningProcesses,
            pynvml.nvmlDeviceGetGraphicsRunningProcesses,
        ):
            try:
                procs = getter(h)
            except pynvml.NVMLError:
                continue
            for p in procs:
                if p.pid in own or p.pid in seen:
                    continue
                try:
                    name = psutil.Process(p.pid).name()
                except psutil.Error:
                    name = "?"
                if name in self.ignore_process_names:
                    continue
                used = getattr(p, "usedGpuMemory", None)
                seen[p.pid] = GpuProcess(p.pid, name, None if used is None else used / MB)
        return {
            "gpu_present": True,
            "vram_total_mb": mem.total / MB,
            "vram_free_mb": mem.free / MB,
            "gpu_util_pct": util,
            "gpu_temp_c": temp,
            "others_on_gpu": tuple(seen.values()),
        }

    def read(self) -> Snapshot:
        own = self.own_pids()
        system_cpu = psutil.cpu_percent(None)
        cpu_others = max(0.0, system_cpu - self._own_cpu_pct(own))
        battery = psutil.sensors_battery()  # None on machines without a battery
        return Snapshot(
            taken_at=time.time(),
            ram_available_mb=psutil.virtual_memory().available / MB,
            cpu_others_pct=cpu_others,
            on_ac=True if battery is None else bool(battery.power_plugged),
            battery_pct=None if battery is None else float(battery.percent),
            **self._gpu(own),
        )


@dataclass
class FakeSensors:
    """Settable sensors for tests and dry runs."""

    snapshot: Snapshot

    def read(self) -> Snapshot:
        return self.snapshot

    def set(self, **changes) -> None:
        from dataclasses import replace

        self.snapshot = replace(self.snapshot, **changes)

    def register_own_pid(self, pid: int) -> None:  # same interface as SystemSensors
        pass

    def unregister_own_pid(self, pid: int) -> None:
        pass
