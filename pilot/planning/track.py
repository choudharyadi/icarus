"""Track model: ordered gates in NED plus vision-based corrections.

The simulator transmits the authoritative course layout (gate poses in NED).
Telemetry position can drift relative to what the camera actually sees (and
PnP gives the truth about where the gate is relative to the vehicle), so each
gate carries a bounded, smoothed correction vector estimated from detections.
"""

import math

import numpy as np

from frames import quat_wxyz_yaw


class PlannedGate:
    def __init__(self, gate_id, pos_ned, yaw_ned, width, height):
        self.gate_id = gate_id
        self.pos_ned = np.asarray(pos_ned, dtype=float)
        self.yaw = yaw_ned
        self.normal = np.array([math.cos(yaw_ned), math.sin(yaw_ned), 0.0])
        self.width = width
        self.height = height
        self.correction = np.zeros(3)

    @property
    def center(self):
        return self.pos_ned + self.correction

    def local_coords(self, p):
        """(u, v, w) = lateral, vertical, along-normal offsets from center."""
        d = np.asarray(p) - self.center
        u_axis = np.array([-self.normal[1], self.normal[0], 0.0])
        return (float(np.dot(d, u_axis)), float(d[2]),
                float(np.dot(d, self.normal)))


class Track:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gates = []

    @property
    def ready(self):
        return len(self.gates) > 0

    def load_from_track_data(self, track_gates):
        if self.gates:
            return
        self.gates = [
            PlannedGate(g.gate_id, g.pos_ned, quat_wxyz_yaw(g.quat_ned),
                        g.width, g.height)
            for g in sorted(track_gates, key=lambda g: g.gate_id)
        ]
        print(f"Track loaded: {len(self.gates)} gates", flush=True)
        for g in self.gates:
            print(f"  gate {g.gate_id}: N{g.pos_ned[0]:+7.2f} "
                  f"E{g.pos_ned[1]:+7.2f} D{g.pos_ned[2]:+6.2f} "
                  f"yaw {math.degrees(g.yaw):+6.1f} deg", flush=True)

    # ------------------------------------------------------------------
    def apply_detection(self, est_center_ned, detection, active_index):
        """Fuse a vision-estimated gate center (NED) into the map.

        Only the active gate and its immediate successor are considered:
        they are the gates that matter and the ones the camera is aimed at.
        Returns the gate id updated, or None.
        """
        pc = self.cfg
        if not pc.correction_enabled or not self.gates:
            return None
        if not pc.correction_min_dist <= detection.distance <= pc.correction_max_dist:
            return None

        candidates = self.gates[active_index:active_index + 2]
        best, best_d = None, pc.association_radius
        for g in candidates:
            d = float(np.linalg.norm(est_center_ned - g.center))
            if d < best_d:
                best, best_d = g, d
        if best is None:
            return None

        offset = est_center_ned - best.pos_ned
        new_corr = ((1.0 - pc.correction_alpha) * best.correction
                    + pc.correction_alpha * offset)
        norm = np.linalg.norm(new_corr)
        if norm > pc.correction_cap:
            new_corr *= pc.correction_cap / norm
        best.correction = new_corr
        return best.gate_id
