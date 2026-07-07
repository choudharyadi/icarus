"""Frame conversion consistency: bridge (Webots<->NED) and pilot (NED native)
must agree, since one encodes telemetry the other decodes.
"""

import math

import numpy as np
import pytest

import frames as bridge_frames  # controllers/dcl_sim_bridge/frames.py

# pilot/frames.py shadows the same module name; import via file path.
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "pilot_frames", os.path.join(ROOT, "pilot", "frames.py"))
pilot_frames = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot_frames)


def test_vec_roundtrip():
    v = np.array([1.0, -2.0, 3.0])
    assert np.allclose(bridge_frames.vec_ned_to_w(bridge_frames.vec_w_to_ned(v)), v)


def test_ned_axes():
    # Webots +X is North, +Y is West (so East = -Y), +Z is Up (Down = -Z).
    assert np.allclose(bridge_frames.vec_w_to_ned([1, 0, 0]), [1, 0, 0])
    assert np.allclose(bridge_frames.vec_w_to_ned([0, 1, 0]), [0, -1, 0])
    assert np.allclose(bridge_frames.vec_w_to_ned([0, 0, 1]), [0, 0, -1])


def test_yaw_conversion():
    # Drone facing Webots +Y (yaw_w = +90 deg) faces NED -90 deg (West... -E).
    assert math.isclose(bridge_frames.yaw_w_to_ned(math.pi / 2), -math.pi / 2)
    assert math.isclose(bridge_frames.yaw_ned_to_w(-math.pi / 2), math.pi / 2)


@pytest.mark.parametrize("yaw_w", [0.0, 0.5, -1.2, 3.0])
def test_pure_yaw_rotmat_to_ned(yaw_w):
    # Webots quaternion for pure yaw about +Z: axis-angle -> xyzw.
    q_w = [0.0, 0.0, math.sin(yaw_w / 2), math.cos(yaw_w / 2)]
    r_w = bridge_frames.quat_xyzw_to_rotmat(q_w)
    r_n = bridge_frames.rotmat_w_to_ned(r_w)
    roll, pitch, yaw = bridge_frames.rotmat_to_euler_ned(r_n)
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert math.isclose(yaw, bridge_frames.wrap_pi(-yaw_w), abs_tol=1e-9)


def test_bridge_euler_matches_pilot_rotmat():
    """Telemetry chain: bridge sends (roll,pitch,yaw); pilot rebuilds the
    rotation with euler_to_rotmat. Both must describe the same rotation."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        # random quaternion
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r_w = bridge_frames.quat_xyzw_to_rotmat([q[0], q[1], q[2], q[3]])
        r_n = bridge_frames.rotmat_w_to_ned(r_w)
        roll, pitch, yaw = bridge_frames.rotmat_to_euler_ned(r_n)
        r_rebuilt = pilot_frames.euler_to_rotmat(roll, pitch, yaw)
        assert np.allclose(r_rebuilt, r_n, atol=1e-9)


def test_bridge_quat_matches_pilot_euler():
    rng = np.random.default_rng(11)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r_w = bridge_frames.quat_xyzw_to_rotmat(list(q))
        r_n = bridge_frames.rotmat_w_to_ned(r_w)
        q_ned = bridge_frames.rotmat_to_quat_wxyz(r_n)
        roll_b, pitch_b, yaw_b = bridge_frames.rotmat_to_euler_ned(r_n)
        roll_p, pitch_p, yaw_p = pilot_frames.quat_wxyz_to_euler(q_ned)
        assert math.isclose(roll_b, roll_p, abs_tol=1e-8)
        assert math.isclose(pitch_b, pitch_p, abs_tol=1e-8)
        assert math.isclose(yaw_b, yaw_p, abs_tol=1e-8)


def test_gate_quat_yaw_roundtrip():
    for yaw in [0.0, 1.0, -2.5, 3.1]:
        q = bridge_frames.yaw_to_quat_ned(yaw)
        assert math.isclose(pilot_frames.quat_wxyz_yaw(q),
                            bridge_frames.wrap_pi(yaw), abs_tol=1e-9)


def test_cam_to_body_axes():
    # Optical axis (z_cam) -> forward and slightly up (negative z in FRD).
    v = pilot_frames.cam_to_body([0, 0, 1])
    assert v[0] > 0.9
    assert abs(v[1]) < 1e-9
    assert v[2] < 0  # up
    assert math.isclose(np.linalg.norm(v), 1.0, abs_tol=1e-9)
    # Image right -> body right.
    assert np.allclose(pilot_frames.cam_to_body([1, 0, 0]), [0, 1, 0])


def test_body_to_ned_level_flight():
    # Facing East (yaw=pi/2), gate 5 m ahead -> 5 m East.
    v = pilot_frames.body_to_ned([5, 0, 0], 0, 0, math.pi / 2)
    assert np.allclose(v, [0, 5, 0], atol=1e-9)
