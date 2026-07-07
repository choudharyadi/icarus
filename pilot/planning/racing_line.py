"""Racing line: gate-threading waypoints, carrot-on-polyline guidance and
speed scheduling.

For every gate three nodes are generated along the (corrected) gate normal:

    approach  A = c - n * approach_dist
    center    C = c
    exit      E = c + n * exit_dist

The drone tracks a carrot point a fixed lookahead ahead of its projection on
the polyline. Segments adjacent to a gate center carry that gate's speed
limit (shaped by how sharp the following turn is); everything else runs at
cruise speed, with a braking profile so each limit is met when it arrives.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    gate_idx: int      # index into track.gates (-1 for the start node)
    kind: str          # "start" | "approach" | "center" | "exit"


@dataclass
class GuidanceCommand:
    carrot: np.ndarray
    speed: float
    look_point: np.ndarray
    gate_idx: int          # gate currently being attacked
    finished: bool


class RacingLine:
    def __init__(self, track, cfg):
        self.track = track
        self.cfg = cfg
        self.nodes = [Node(-1, "start")]
        for i in range(len(track.gates)):
            self.nodes.append(Node(i, "approach"))
            self.nodes.append(Node(i, "center"))
            self.nodes.append(Node(i, "exit"))
        self.seg = 0                      # current segment = nodes[seg]->nodes[seg+1]
        self.start_point = None

    # ------------------------------------------------------------------
    def begin(self, start_pos):
        self.start_point = np.asarray(start_pos, dtype=float).copy()
        self.seg = 0

    def _point(self, node):
        if node.kind == "start":
            return self.start_point
        g = self.track.gates[node.gate_idx]
        if node.kind == "approach":
            return g.center - g.normal * self.cfg.approach_dist
        if node.kind == "exit":
            return g.center + g.normal * self.cfg.exit_dist
        return g.center

    def _points(self):
        return [self._point(n) for n in self.nodes]

    # ------------------------------------------------------------------
    def _gate_speed_limit(self, gate_idx):
        """Speed allowed while threading gate `gate_idx`, shaped by the
        heading change required to line up the following gate."""
        gates = self.track.gates
        v = self.cfg.v_gate
        if gate_idx + 1 < len(gates):
            g, gn = gates[gate_idx], gates[gate_idx + 1]
            cosang = float(np.clip(np.dot(g.normal, gn.normal), -1.0, 1.0))
            theta = math.acos(cosang)
            v = v * max(0.45, 1.0 - 0.35 * self.cfg.turn_speed_k
                        * min(theta, 1.6))
        return max(self.cfg.v_min, v)

    def _segment_limit(self, seg_idx):
        """Speed limit applying over segment nodes[seg_idx] -> nodes[seg_idx+1]."""
        a, b = self.nodes[seg_idx], self.nodes[seg_idx + 1]
        if b.kind in ("center",):
            return self._gate_speed_limit(b.gate_idx)
        if a.kind in ("center",):
            return self._gate_speed_limit(a.gate_idx)
        return self.cfg.v_cruise

    # ------------------------------------------------------------------
    def notify_gate_passed(self, next_gate_index):
        """Race status reports gates 0..next_gate_index-1 passed: never track
        a segment before the exit of the last passed gate."""
        if next_gate_index <= 0:
            return
        # Segment index of C_{k-1} -> E_{k-1}:  start node + 3 per gate.
        min_seg = 3 * (next_gate_index - 1) + 2
        min_seg = min(min_seg, len(self.nodes) - 2)
        if self.seg < min_seg:
            self.seg = min_seg

    # ------------------------------------------------------------------
    def command(self, pos):
        pos = np.asarray(pos, dtype=float)
        pts = self._points()
        n_seg = len(pts) - 1

        # ---- advance progress by projection -----------------------------
        while True:
            a, b = pts[self.seg], pts[self.seg + 1]
            ab = b - a
            ab_len2 = float(np.dot(ab, ab))
            t = 0.0 if ab_len2 < 1e-9 else float(np.dot(pos - a, ab) / ab_len2)
            if t >= 1.0 and self.seg < n_seg - 1:
                self.seg += 1
                continue
            break
        t = max(0.0, min(1.0, t))
        proj = a + t * ab

        # ---- walk the carrot forward ------------------------------------
        # Carrot grows with planned speed: tighter tracking through gates,
        # smoother cutting on fast sections.
        seg_limit_now = self._segment_limit(self.seg)
        remaining = min(self.cfg.lookahead_max,
                        max(self.cfg.lookahead,
                            self.cfg.lookahead_speed_gain * seg_limit_now
                            + 0.6))
        carrot = proj
        seg_i, local = self.seg, t
        while remaining > 0.0:
            a_i, b_i = pts[seg_i], pts[seg_i + 1]
            seg_vec = b_i - a_i
            seg_len = float(np.linalg.norm(seg_vec))
            left = (1.0 - local) * seg_len
            if remaining <= left or seg_i >= n_seg - 1:
                frac = 0.0 if seg_len < 1e-9 else min(
                    1.0, local + remaining / seg_len)
                carrot = a_i + frac * seg_vec
                break
            remaining -= left
            seg_i += 1
            local = 0.0

        # ---- speed scheduling --------------------------------------------
        current_limit = self._segment_limit(self.seg)
        speed = current_limit
        # Brake ahead of any slower constraint within braking range.
        dist_acc = float(np.linalg.norm(pts[self.seg + 1] - proj))
        j = self.seg + 1
        while j < n_seg and dist_acc < 25.0:
            lim = self._segment_limit(j)
            if lim < speed:
                allowed = math.sqrt(lim * lim + 2.0 * self.cfg.decel * dist_acc)
                speed = min(speed, max(lim, allowed))
            dist_acc += float(np.linalg.norm(pts[j + 1] - pts[j]))
            j += 1
        # Come to a stop at the end of the plan.
        dist_to_end = float(np.linalg.norm(pts[self.seg + 1] - proj))
        for k in range(self.seg + 1, n_seg):
            dist_to_end += float(np.linalg.norm(pts[k + 1] - pts[k]))
        stop_allowed = math.sqrt(max(0.0, 2.0 * self.cfg.decel
                                     * (dist_to_end + self.cfg.finish_brake_dist - 1.0)))
        speed = max(self.cfg.v_min if dist_to_end > 0.4 else 0.0,
                    min(speed, stop_allowed))

        # ---- yaw look point -----------------------------------------------
        # Aim the camera at the next gate still to be crossed: the first
        # "center" node at or beyond the current segment end.
        gates = self.track.gates
        attack_gate = len(gates) - 1
        for idx in range(self.seg + 1, len(self.nodes)):
            if self.nodes[idx].kind == "center":
                attack_gate = self.nodes[idx].gate_idx
                break
        g = gates[attack_gate]
        d_to_gate = float(np.linalg.norm(g.center - pos))
        if d_to_gate < self.cfg.yaw_hold_dist:
            look_point = g.center + g.normal * 3.0
        else:
            look_point = g.center

        finished = self.seg >= n_seg - 1 and t >= 1.0 - 1e-6
        return GuidanceCommand(
            carrot=carrot, speed=float(speed), look_point=look_point,
            gate_idx=attack_gate, finished=finished)
