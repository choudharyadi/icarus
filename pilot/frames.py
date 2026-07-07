"""Frame math for the pilot. Everything is NED / body-FRD / OpenCV-camera.

NED:    x North, y East, z Down (MAV_FRAME_LOCAL_NED, origin at arming point)
FRD:    x forward, y right, z down (vehicle body)
Camera: OpenCV convention - x right, y down, z forward (optical axis);
        mounted at the body origin, pitched 20 degrees up (spec 3.8).
"""

import math

import numpy as np

CAMERA_PITCH_UP = math.radians(20.0)


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def euler_to_rotmat(roll, pitch, yaw):
    """ZYX aerospace Euler -> rotation matrix (body-FRD -> NED)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def quat_wxyz_to_euler(q):
    """Quaternion (w, x, y, z) -> (roll, pitch, yaw) aerospace ZYX."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, s)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_wxyz_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# Camera (OpenCV axes) -> body FRD, for a camera pitched up by CAMERA_PITCH_UP.
_ct, _st = math.cos(CAMERA_PITCH_UP), math.sin(CAMERA_PITCH_UP)
R_BODY_CAM = np.array([
    [0.0, _st, _ct],   # body x (forward)
    [1.0, 0.0, 0.0],   # body y (right)
    [0.0, _ct, -_st],  # body z (down)
])


def cam_to_body(v_cam):
    """OpenCV camera vector -> body FRD vector."""
    return R_BODY_CAM @ np.asarray(v_cam, dtype=float)


def body_to_ned(v_body, roll, pitch, yaw):
    return euler_to_rotmat(roll, pitch, yaw) @ np.asarray(v_body, dtype=float)


def bearing_to(p_from, p_to):
    """NED yaw angle pointing from p_from to p_to."""
    d = np.asarray(p_to) - np.asarray(p_from)
    return math.atan2(d[1], d[0])
