from dataclasses import replace
from datetime import datetime

import pytest

from engine.policy import Policy, PolicyConfig, State, in_window
from engine.sensors import GpuProcess, Snapshot

DAYTIME = datetime(2026, 11, 23, 14, 0)
NIGHT = datetime(2026, 11, 23, 2, 0)


def snap(**kw) -> Snapshot:
    base = Snapshot(taken_at=1000.0, gpu_present=True, vram_total_mb=6141, vram_free_mb=6000,
                    gpu_temp_c=55, ram_available_mb=8000, cpu_others_pct=5, on_ac=True)
    return replace(base, **kw)


def decide(policy, s, now=DAYTIME, pressure=0.0, current=State.REST, in_state=1e9, day_work=True):
    return policy.decide(s, now, pressure, current, in_state, day_work)


def test_night_window_crosses_midnight():
    assert in_window(NIGHT, "23:00", "07:00")
    assert in_window(datetime(2026, 1, 1, 23, 30), "23:00", "07:00")
    assert not in_window(DAYTIME, "23:00", "07:00")
    assert in_window(DAYTIME, "13:00", "15:00")


def test_day_with_new_material_reads():
    d = decide(Policy(), snap())
    assert d.state is State.DAY and not d.hard_stop and d.budget > 0


def test_day_without_material_rests():
    d = decide(Policy(), snap(), day_work=False)
    assert d.state is State.REST and "nothing new" in d.reason


def test_other_program_on_gpu_is_a_hard_stop_with_hysteresis():
    p = Policy()
    game = (GpuProcess(4242, "game", 3000.0),)
    d = decide(p, snap(others_on_gpu=game), current=State.DAY, in_state=0)
    assert d.state is State.REST and d.hard_stop and "game" in d.reason
    # Game shrinks to 300 MB: still above the 200 MB resume line, so keep resting.
    d = decide(p, snap(others_on_gpu=(GpuProcess(4242, "game", 300.0),)), current=State.REST)
    assert d.state is State.REST
    # Game gone: resume immediately (leaving REST is never delayed).
    d = decide(p, snap(), current=State.REST, in_state=0)
    assert d.state is State.DAY


def test_temperature_hysteresis():
    p = Policy()
    assert decide(p, snap(gpu_temp_c=84)).state is State.REST
    assert decide(p, snap(gpu_temp_c=78)).state is State.REST  # cooling, but not below 75 yet
    assert decide(p, snap(gpu_temp_c=74)).state is State.DAY


def test_low_ram_and_battery():
    assert "RAM" in decide(Policy(), snap(ram_available_mb=1500)).reason
    d = decide(Policy(), snap(on_ac=False, battery_pct=60))
    assert d.state is State.REST and "battery" in d.reason
    cfg = PolicyConfig(allow_on_battery=True)
    assert decide(Policy(cfg), snap(on_ac=False)).state is State.DAY


def test_no_gpu():
    d = decide(Policy(), snap(gpu_present=False))
    assert d.state is State.REST and "no NVIDIA GPU" in d.reason


def test_night_dreams_then_sleeps_when_pressure_is_high():
    p = Policy(PolicyConfig(night_idle_seconds=0))
    assert decide(p, snap(), now=NIGHT, pressure=0.3).state is State.DREAM
    assert decide(p, snap(), now=NIGHT, pressure=1.2).state is State.DEEP_SLEEP
    # Once asleep, it stays asleep until the pressure is fully relieved...
    assert decide(p, snap(), now=NIGHT, pressure=0.4, current=State.DEEP_SLEEP).state is State.DEEP_SLEEP
    # ...then goes back to dreaming.
    assert decide(p, snap(), now=NIGHT, pressure=0.0, current=State.DEEP_SLEEP).state is State.DREAM


def test_sleep_needs_more_gpu_memory_than_dreaming():
    p = Policy(PolicyConfig(night_idle_seconds=0))
    small = snap(others_on_gpu=(GpuProcess(1, "browser", 150.0),), vram_total_mb=5000)
    d = decide(p, small, now=NIGHT, pressure=2.0)
    assert d.state is State.DREAM and "not enough GPU memory to train" in d.reason


def test_nap_after_long_idle_by_day():
    p = Policy(PolicyConfig(nap_after_idle_seconds=600))
    assert decide(p, snap(taken_at=0), day_work=False).state is State.REST
    d = decide(p, snap(taken_at=700), day_work=False)
    assert d.state is State.DREAM and "nap" in d.reason
    # Someone starts working: idle resets, nap ends (after min time; in_state is large here).
    assert decide(p, snap(taken_at=800, cpu_others_pct=80), current=State.DREAM,
                  day_work=False).state is State.REST


def test_min_time_in_state_prevents_flapping_but_not_hard_stops():
    p = Policy(PolicyConfig(night_idle_seconds=0, min_state_seconds=120))
    d = decide(p, snap(), now=NIGHT, current=State.DAY, in_state=30)
    assert d.state is State.DAY and "staying" in d.reason
    d = decide(p, snap(gpu_temp_c=90), now=NIGHT, current=State.DAY, in_state=30)
    assert d.state is State.REST and d.hard_stop


def test_budget_shrinks_as_the_machine_gets_busy():
    p = Policy()
    calm = decide(p, snap()).budget
    busy = decide(p, snap(cpu_others_pct=25, gpu_temp_c=78)).budget
    assert 0 <= busy < calm <= 1


def test_config_from_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[engine]\nnight_start = "22:00"\nallow_on_battery = true\n')
    cfg = PolicyConfig.from_toml(path)
    assert cfg.night_start == "22:00" and cfg.allow_on_battery
    path.write_text("[engine]\nnot_a_setting = 1\n")
    with pytest.raises(ValueError):
        PolicyConfig.from_toml(path)
    assert PolicyConfig.from_toml(tmp_path / "missing.toml") == PolicyConfig()
