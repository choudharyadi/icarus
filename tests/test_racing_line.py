"""Racing line geometry and speed scheduling."""

import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot_frames = _load("pilot_frames", "pilot/frames.py")
sys.modules["frames"] = pilot_frames
config = _load("pilot_config", "pilot/config.py")
track_mod = _load("pilot_track", "pilot/planning/track.py")
racing_line = _load("pilot_racing_line", "pilot/planning/racing_line.py")


def straight_track(n=3, spacing=8.0, depth=-2.0):
    """Gates straight ahead along North."""
    t = track_mod.Track(config.PerceptionConfig())
    t.gates = [
        track_mod.PlannedGate(i, [spacing * (i + 1), 0.0, depth], 0.0, 1.5, 1.5)
        for i in range(n)
    ]
    return t


def test_carrot_progresses_and_finishes():
    t = straight_track()
    rc = config.RaceConfig()
    line = racing_line.RacingLine(t, rc)
    line.begin([0.0, 0.0, -2.0])

    pos = np.array([0.0, 0.0, -2.0])
    last_n = -1.0
    finished = False
    for _ in range(3000):
        cmd = line.command(pos)
        d = cmd.carrot - pos
        dist = np.linalg.norm(d)
        if dist > 1e-6:
            pos = pos + d / dist * min(dist, cmd.speed * 0.05)
        assert cmd.speed <= rc.v_cruise + 1e-6
        if cmd.finished:
            finished = True
            break
        assert pos[0] >= last_n - 0.5
        last_n = pos[0]
    assert finished
    # Ended past the last gate's exit point.
    assert pos[0] > t.gates[-1].center[0] + rc.exit_dist - 0.5
    assert abs(pos[1]) < 0.3


def test_gate_zone_speed_limited():
    t = straight_track()
    rc = config.RaceConfig()
    line = racing_line.RacingLine(t, rc)
    line.begin([0.0, 0.0, -2.0])
    # Standing just before gate 0's approach point.
    cmd = line.command(np.array([t.gates[0].center[0] - rc.approach_dist - 0.1,
                                 0.0, -2.0]))
    assert cmd.speed <= rc.v_gate + 0.4  # braking into the gate zone


def test_turn_shapes_gate_speed():
    rc = config.RaceConfig()
    t_straight = straight_track()
    line_s = racing_line.RacingLine(t_straight, rc)
    v_straight = line_s._gate_speed_limit(0)

    t_turn = straight_track()
    # Make gate 1 require a hard 60 degree heading change after gate 0.
    t_turn.gates[1] = track_mod.PlannedGate(
        1, [16.0, 0.0, -2.0], math.radians(60.0), 1.5, 1.5)
    line_t = racing_line.RacingLine(t_turn, rc)
    assert line_t._gate_speed_limit(0) < v_straight


def test_notify_gate_passed_clamps_progress():
    t = straight_track()
    rc = config.RaceConfig()
    line = racing_line.RacingLine(t, rc)
    line.begin([0.0, 0.0, -2.0])
    assert line.seg == 0
    line.notify_gate_passed(2)  # gates 0 and 1 passed
    # Must be at least at segment C_1 -> E_1: index 3*(2-1)+2 = 5.
    assert line.seg >= 5


def test_look_point_targets_next_gate():
    t = straight_track()
    rc = config.RaceConfig()
    line = racing_line.RacingLine(t, rc)
    line.begin([0.0, 0.0, -2.0])
    cmd = line.command(np.array([0.0, 0.0, -2.0]))
    assert cmd.gate_idx == 0
    assert np.allclose(cmd.look_point[:2], t.gates[0].center[:2], atol=1e-6)


def test_correction_capped():
    pc = config.PerceptionConfig()
    t = straight_track()
    t.cfg = pc
    gate = t.gates[0]

    class Det:
        distance = 5.0

    # Huge bogus offset should be capped.
    for _ in range(50):
        t.apply_detection(gate.pos_ned + np.array([2.5, 2.5, 0.0]), Det(), 0)
    assert np.linalg.norm(gate.correction) <= pc.correction_cap + 1e-9


def test_correction_rejects_far_association():
    pc = config.PerceptionConfig()
    t = straight_track()
    t.cfg = pc

    class Det:
        distance = 5.0

    updated = t.apply_detection(np.array([100.0, 50.0, -2.0]), Det(), 0)
    assert updated is None
