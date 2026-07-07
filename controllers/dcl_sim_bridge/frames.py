"""Coordinate-frame conversions between Webots world/body frames and MAVLink NED frames.

Conventions
-----------
Webots world (W):  right-handed, Z up.  Drone body in Webots is FLU
                   (x forward, y left, z up).
MAVLink NED (N):   N = +X_w, E = -Y_w, D = -Z_w.
Body FRD:          x forward, y right, z down  (FRD = C @ FLU).

The single change-of-basis matrix C = diag(1, -1, -1) maps both world ENU->NED
and body FLU->FRD.  For a rotation matrix R_w (body-FLU -> world-W) the NED
equivalent (body-FRD -> NED) is  R_n = C @ R_w @ C.
"""

import math

import numpy as np

# Change of basis (its own inverse/transpose).
C = np.diag([1.0, -1.0, -1.0])


def vec_w_to_ned(v):
    """Webots world vector -> NED vector."""
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def vec_ned_to_w(v):
    """NED vector -> Webots world vector (same involution)."""
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def body_flu_to_frd(v):
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def body_frd_to_flu(v):
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def yaw_w_to_ned(yaw_w):
    """Webots yaw (CCW about +Z_w from +X_w) -> NED yaw (CW about D from North)."""
    return wrap_pi(-yaw_w)


def yaw_ned_to_w(yaw_ned):
    return wrap_pi(-yaw_ned)


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_xyzw_to_rotmat(q):
    """Webots InertialUnit quaternion [x, y, z, w] -> rotation matrix (body-FLU -> world)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_w_to_ned(r_w):
    """Rotation (body-FLU -> world-W)  ->  rotation (body-FRD -> NED)."""
    return C @ r_w @ C


def rotmat_to_euler_ned(r_n):
    """Aerospace ZYX Euler angles (roll, pitch, yaw) from a body-FRD -> NED rotation."""
    pitch = -math.asin(max(-1.0, min(1.0, r_n[2, 0])))
    roll = math.atan2(r_n[2, 1], r_n[2, 2])
    yaw = math.atan2(r_n[1, 0], r_n[0, 0])
    return roll, pitch, yaw


def rotmat_to_quat_wxyz(r):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def yaw_to_quat_ned(yaw_ned):
    """Pure-yaw NED quaternion (w, x, y, z)."""
    return np.array([math.cos(yaw_ned / 2.0), 0.0, 0.0, math.sin(yaw_ned / 2.0)])
