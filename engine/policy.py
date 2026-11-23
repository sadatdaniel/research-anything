"""The decision rules: which state the learner should be in right now, and why.

Two drives, borrowed from Borbély's two-process model of human sleep:
  * sleep pressure  - rises with each confirmed finding not yet trained in, falls during deep sleep
  * machine budget  - what the hardware can spare right now (GPU memory, heat, RAM, power, hours)

Hard stops (someone else needs the GPU, too hot, low RAM, on battery) always win and
send the learner to REST immediately. Everything else respects a minimum time per state
so the learner does not flap between states (loading the model costs 10-30 s).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from datetime import datetime, time as dtime
from enum import Enum
from pathlib import Path

from engine.sensors import Snapshot


class State(str, Enum):
    DAY = "day"  # read new material, take notes, self-quiz (light)
    DREAM = "dream"  # hunt for connections between distant notes and test them (medium-high)
    DEEP_SLEEP = "deep_sleep"  # train confirmed findings into the adapter (heaviest)
    REST = "rest"  # model unloaded, GPU handed back


@dataclass
class PolicyConfig:
    night_start: str = "23:00"
    night_end: str = "07:00"
    allow_on_battery: bool = False
    # Hard-stop thresholds, each with a lower "resume" value so it doesn't flap.
    gpu_temp_stop_c: float = 83.0
    gpu_temp_resume_c: float = 75.0
    ram_stop_mb: float = 2048.0
    ram_resume_mb: float = 2560.0
    others_vram_stop_mb: float = 500.0
    others_vram_resume_mb: float = 200.0
    # GPU memory each state needs (total minus what other programs hold). Measured in Phase 0 with
    # Ouro-1.4B: batch-4 inference ~3.9 GB per process, LoRA training at 1024 tokens ~5.0 GB.
    vram_need_day_mb: float = 4000.0
    vram_need_dream_mb: float = 4000.0
    vram_need_deep_sleep_mb: float = 5200.0
    # Idle detection, from CPU use by other programs.
    idle_cpu_pct: float = 30.0
    night_idle_seconds: float = 60.0  # at night, idle this long before dreaming
    nap_after_idle_seconds: float = 20 * 60.0  # by day, idle this long before a nap
    # Sleep pressure: enter deep sleep at or above this, stay until it falls to zero.
    sleep_pressure_threshold: float = 1.0
    min_state_seconds: float = 120.0
    ignore_gpu_process_names: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown engine settings: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_toml(cls, path: str | Path) -> "PolicyConfig":
        """Read the [engine] table of a config.toml; a missing file gives the defaults."""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("rb") as fh:
            return cls.from_dict(tomllib.load(fh).get("engine", {}))

    def vram_need(self, state: State) -> float:
        return {
            State.DAY: self.vram_need_day_mb,
            State.DREAM: self.vram_need_dream_mb,
            State.DEEP_SLEEP: self.vram_need_deep_sleep_mb,
            State.REST: 0.0,
        }[state]


@dataclass(frozen=True)
class Decision:
    state: State
    reason: str
    budget: float  # 0..1, how much headroom there is; the workload scales loops and session length by it
    hard_stop: bool = False


def _parse_hhmm(text: str) -> dtime:
    hours, minutes = text.split(":")
    return dtime(int(hours), int(minutes))


def in_window(now: datetime, start: str, end: str) -> bool:
    """True if now's clock time is inside [start, end); handles windows that cross midnight."""
    t, s, e = now.time(), _parse_hhmm(start), _parse_hhmm(end)
    return s <= t < e if s <= e else (t >= s or t < e)


class Policy:
    """Stateful only for hysteresis latches and idle tracking; otherwise a pure function of its inputs."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self._latched: dict[str, str] = {}  # latch name -> human-readable reason
        self._idle_since: float | None = None

    # ---- hard stops -------------------------------------------------------------------------

    def _update_latch(self, name: str, set_when: bool, clear_when: bool, reason: str) -> None:
        if set_when:
            self._latched[name] = reason
        elif clear_when:
            self._latched.pop(name, None)
        elif name in self._latched:
            self._latched[name] = reason  # still latched: refresh the numbers in the reason

    def hard_stops(self, snap: Snapshot) -> list[str]:
        """Reasons the learner must rest right now (empty list = none). Updates the latches."""
        c = self.config
        if not snap.gpu_present:
            return ["no NVIDIA GPU found"]
        others = snap.others_vram_mb
        names = ", ".join(sorted({p.name for p in snap.others_on_gpu})) or "another program"
        self._update_latch(
            "others",
            others >= c.others_vram_stop_mb,
            others <= c.others_vram_resume_mb,
            f"{names} using {others / 1024:.1f} GB of GPU memory",
        )
        self._update_latch(
            "temp",
            snap.gpu_temp_c >= c.gpu_temp_stop_c,
            snap.gpu_temp_c < c.gpu_temp_resume_c,
            f"GPU at {snap.gpu_temp_c:.0f} °C (resumes below {c.gpu_temp_resume_c:.0f} °C)",
        )
        self._update_latch(
            "ram",
            snap.ram_available_mb < c.ram_stop_mb,
            snap.ram_available_mb > c.ram_resume_mb,
            f"only {snap.ram_available_mb / 1024:.1f} GB of RAM free",
        )
        reasons = list(self._latched.values())
        if not snap.on_ac and not c.allow_on_battery:
            pct = "" if snap.battery_pct is None else f" ({snap.battery_pct:.0f}%)"
            reasons.append(f"on battery{pct}")
        return reasons

    # ---- idle tracking ----------------------------------------------------------------------

    def _idle_seconds(self, snap: Snapshot) -> float:
        idle_now = (
            snap.cpu_others_pct < self.config.idle_cpu_pct
            and snap.others_vram_mb <= self.config.others_vram_resume_mb
        )
        if not idle_now:
            self._idle_since = None
            return 0.0
        if self._idle_since is None:
            self._idle_since = snap.taken_at
        return snap.taken_at - self._idle_since

    # ---- budget -----------------------------------------------------------------------------

    def budget(self, snap: Snapshot, state: State) -> float:
        if state is State.REST:
            return 0.0
        c = self.config
        need = c.vram_need(state)
        vram = (snap.vram_for_us_mb - need) / need if need else 1.0
        temp = (c.gpu_temp_stop_c - snap.gpu_temp_c) / max(1.0, c.gpu_temp_stop_c - 50.0)
        cpu = 1.0 - snap.cpu_others_pct / 100.0
        return round(max(0.0, min(1.0, vram * 4, temp, cpu)), 3)

    # ---- the decision -----------------------------------------------------------------------

    def _candidate(self, snap: Snapshot, now: datetime, pressure: float, current: State,
                   has_day_work: bool, has_dream_work: bool) -> tuple[State, str]:
        c = self.config
        idle = self._idle_seconds(snap)
        night = in_window(now, c.night_start, c.night_end)
        sleeping_time = (night and idle >= c.night_idle_seconds) or (not night and idle >= c.nap_after_idle_seconds)
        room = snap.vram_for_us_mb

        if sleeping_time:
            label = "night" if night else f"nap: idle {idle / 60:.0f} min"
            wants_sleep = pressure >= c.sleep_pressure_threshold or (current is State.DEEP_SLEEP and pressure > 0)
            if wants_sleep and room >= c.vram_need_deep_sleep_mb:
                return State.DEEP_SLEEP, f"{label}; sleep pressure {pressure:.2f}, {room / 1024:.1f} GB free for training"
            # You can't dream about what you haven't read: unread material comes first, even at night.
            if has_day_work and room >= c.vram_need_day_mb:
                return State.DAY, f"{label}; reading new material first"
            if not has_dream_work:
                return State.REST, f"{label}; nothing to dream about yet"
            if room >= c.vram_need_dream_mb:
                extra = f"; sleep pressure {pressure:.2f} but not enough GPU memory to train" if wants_sleep else ""
                return State.DREAM, f"{label}; hunting for connections{extra}"
            return State.REST, f"{label}, but only {room / 1024:.1f} GB of GPU memory available"

        if not has_day_work:
            return State.REST, "nothing new to read; waiting for night or a nap"
        if room >= c.vram_need_day_mb:
            return State.DAY, "new material to read"
        return State.REST, f"new material waiting, but only {room / 1024:.1f} GB of GPU memory available"

    def _still_valid(self, state: State, snap: Snapshot, has_day_work: bool, has_dream_work: bool) -> bool:
        if state is State.REST:
            return False  # leaving REST is never delayed; hard stops have their own hysteresis
        if (state is State.DAY and not has_day_work) or (state is State.DREAM and not has_dream_work):
            return False
        return snap.vram_for_us_mb >= self.config.vram_need(state)

    def decide(self, snap: Snapshot, now: datetime, pressure: float, current: State,
               seconds_in_state: float, has_day_work: bool, has_dream_work: bool = True) -> Decision:
        stops = self.hard_stops(snap)
        if stops:
            self._idle_since = None
            return Decision(State.REST, "; ".join(stops), 0.0, hard_stop=True)

        state, reason = self._candidate(snap, now, pressure, current, has_day_work, has_dream_work)
        if (
            state is not current
            and seconds_in_state < self.config.min_state_seconds
            and self._still_valid(current, snap, has_day_work, has_dream_work)
        ):
            wait = self.config.min_state_seconds - seconds_in_state
            return Decision(current, f"staying {wait:.0f} s more before switching to {state.value}",
                            self.budget(snap, current))
        return Decision(state, reason, self.budget(snap, state))
