"""Gate detector accuracy on synthetically rendered gates.

Renders the competition gate (2.7 m outer / 1.5 m inner red frame) with the
exact spec camera model, then checks the PnP range/bearing recovery.
"""

import importlib.util
import os

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load pilot modules under unambiguous names.
pilot_frames = _load("pilot_frames", "pilot/frames.py")
import sys
sys.modules["frames"] = pilot_frames  # gate_detector does `from frames import`
config = _load("pilot_config", "pilot/config.py")
gate_detector = _load("pilot_gate_detector", "pilot/perception/gate_detector.py")

CFG = config.PerceptionConfig()
BG = (190, 200, 205)  # sandy-ish background
RED = (30, 30, 220)   # BGR


def render_gate(tvec, rvec=(0.0, 0.0, 0.0)):
    """Render outer red frame with inner cut-out at the given camera-frame
    pose (OpenCV axes)."""
    cam = np.array([[CFG.fx, 0, CFG.cx], [0, CFG.fy, CFG.cy], [0, 0, 1]],
                   dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    img = np.full((CFG.image_height, CFG.image_width, 3), BG, dtype=np.uint8)

    half_o = 2.7 / 2.0
    half_i = 1.5 / 2.0
    outer = np.array([[-half_o, -half_o, 0], [half_o, -half_o, 0],
                      [half_o, half_o, 0], [-half_o, half_o, 0]],
                     dtype=np.float32)
    inner = np.array([[-half_i, -half_i, 0], [half_i, -half_i, 0],
                      [half_i, half_i, 0], [-half_i, half_i, 0]],
                     dtype=np.float32)
    rvec = np.array(rvec, dtype=np.float32)
    tvec_a = np.array(tvec, dtype=np.float32)
    outer_px, _ = cv2.projectPoints(outer, rvec, tvec_a, cam, dist)
    inner_px, _ = cv2.projectPoints(inner, rvec, tvec_a, cam, dist)
    cv2.fillPoly(img, [np.int32(outer_px.reshape(-1, 2))], RED)
    cv2.fillPoly(img, [np.int32(inner_px.reshape(-1, 2))], BG)
    return img


@pytest.mark.parametrize("tvec", [
    (0.0, 0.0, 5.0),
    (0.0, 0.0, 10.0),
    (1.0, -0.5, 7.0),
    (-1.5, 0.6, 8.0),
])
def test_frontal_gate_position(tvec):
    detector = gate_detector.GateDetector(CFG)
    img = render_gate(tvec)
    dets = detector.detect(img)
    assert len(dets) == 1, f"expected 1 detection, got {len(dets)}"
    est = dets[0].pos_cam
    err = np.linalg.norm(est - np.array(tvec))
    # Corner quantization grows with range: allow ~5% of distance.
    tol = max(0.25, 0.05 * tvec[2])
    assert err < tol, f"pose error {err:.3f} m for gate at {tvec}"


def test_angled_gate_detected():
    detector = gate_detector.GateDetector(CFG)
    img = render_gate((0.8, 0.0, 7.0), rvec=(0.0, 0.45, 0.0))
    dets = detector.detect(img)
    assert len(dets) == 1
    # Range should still be close even with yawed gate.
    assert abs(dets[0].pos_cam[2] - 7.0) < 0.8


def test_two_gates():
    detector = gate_detector.GateDetector(CFG)
    img = render_gate((-2.2, 0.0, 9.0))
    # second, nearer gate to the right
    img2 = render_gate((2.0, 0.2, 6.0))
    mask = np.any(img2 != np.array(BG, dtype=np.uint8), axis=2)
    img[mask] = img2[mask]
    dets = detector.detect(img)
    assert len(dets) == 2
    # Strongest (largest) first = the nearer gate.
    assert dets[0].distance < dets[1].distance


def test_no_false_positive_on_empty():
    detector = gate_detector.GateDetector(CFG)
    img = np.full((CFG.image_height, CFG.image_width, 3), BG, dtype=np.uint8)
    assert detector.detect(img) == []


def test_body_frame_tilt_compensation():
    """A gate dead-center in the image sits 20 degrees *above* the body
    forward axis (camera is pitched up)."""
    detector = gate_detector.GateDetector(CFG)
    img = render_gate((0.0, 0.0, 6.0))
    det = detector.detect(img)[0]
    body = det.pos_body
    assert body[0] > 5.0          # forward
    assert abs(body[1]) < 0.2     # centred
    assert body[2] < -1.5         # above (negative down)
