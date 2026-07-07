"""Race management: gate discovery, passage detection, timing, collisions.

Runs inside the supervisor-enabled bridge controller. The world's QualifierGate
nodes define the course; gates are ordered by the numeric suffix of their
`name` field ("gate0", "gate1", ...). Gate 0 is the start gate, the last gate
is the finish gate.
"""

import math
import re

import numpy as np

import wire
from frames import vec_w_to_ned, yaw_to_quat_ned

GATE_INNER_HALF = 0.75      # 1.5 m inner opening
GATE_PLANE_HALF_DEPTH = 0.13


class Gate:
    def __init__(self, index, node, pos_w, yaw_w):
        self.index = index
        self.node = node
        self.pos_w = np.asarray(pos_w, dtype=float)
        self.yaw_w = yaw_w
        # Local axes in world frame: u = width axis, n = travel normal.
        self.u_w = np.array([math.cos(yaw_w), math.sin(yaw_w), 0.0])
        self.n_w = np.array([-math.sin(yaw_w), math.cos(yaw_w), 0.0])

    def local_coords(self, p_w):
        """(u, v, w): width offset, height offset, signed distance along
        travel normal."""
        d = p_w - self.pos_w
        return (float(np.dot(d, self.u_w)), float(d[2]),
                float(np.dot(d, self.n_w)))

    def to_track_record(self, origin_w):
        pos_ned = vec_w_to_ned(self.pos_w - origin_w)
        d_ned = vec_w_to_ned(self.n_w)
        yaw_ned = math.atan2(d_ned[1], d_ned[0])
        return {
            "id": self.index,
            "pos_ned": pos_ned,
            "quat_ned": yaw_to_quat_ned(yaw_ned),
            "width": 2 * GATE_INNER_HALF,
            "height": 2 * GATE_INNER_HALF,
        }


class RaceManager:
    def __init__(self, supervisor, origin_w, max_duration_s=480.0):
        self.supervisor = supervisor
        self.origin_w = np.asarray(origin_w, dtype=float)
        self.max_duration_s = max_duration_s
        self.gates = self._discover_gates()
        self.reset()
        names = ", ".join(f"gate{g.index}" for g in self.gates)
        print(f"[RACE] Course loaded: {len(self.gates)} gates ({names})")

    # ------------------------------------------------------------------
    def _discover_gates(self):
        gates = []
        root_children = self.supervisor.getRoot().getField("children")
        for i in range(root_children.getCount()):
            node = root_children.getMFNode(i)
            try:
                if node.getTypeName() != "QualifierGate":
                    continue
            except Exception:
                continue
            name_field = node.getField("name")
            name = name_field.getSFString() if name_field else ""
            m = re.search(r"(\d+)\s*$", name)
            if not m:
                continue
            idx = int(m.group(1))
            pos = node.getField("translation").getSFVec3f()
            rot = node.getField("rotation").getSFRotation()
            # Axis-angle, assumed about +/-Z.
            yaw = rot[3] * (1.0 if rot[2] >= 0 else -1.0)
            gates.append(Gate(idx, node, pos, yaw))
        gates.sort(key=lambda g: g.index)
        # Re-index densely in course order.
        for order, g in enumerate(gates):
            g.index = order
        return gates

    def reset(self):
        self.active_gate_index = 0
        self.race_start_time = None     # sim seconds
        self.race_finish_time = None    # sim seconds
        self.last_gate_race_time = None
        self.gate_split_times = []
        self._prev_w = None
        self._prev_pos = None
        self._last_collision_time = -1e9
        self._announced_finish = False

    # ------------------------------------------------------------------
    def track_records(self):
        return [g.to_track_record(self.origin_w) for g in self.gates]

    def update(self, sim_time, drone_pos_w):
        """Advance race state. Returns list of console events (strings)."""
        events = []
        if not self.gates or self.race_finish_time is not None:
            return events

        gate = self.gates[self.active_gate_index]
        u, v, w = gate.local_coords(np.asarray(drone_pos_w))

        if self._prev_w is not None and self._prev_w < 0.0 <= w:
            # Crossed the gate plane in travel direction; interpolate the
            # crossing point for an accurate in-opening check.
            frac = -self._prev_w / max(1e-9, (w - self._prev_w))
            cross = self._prev_pos + frac * (np.asarray(drone_pos_w) - self._prev_pos)
            cu, cv, _ = gate.local_coords(cross)
            if abs(cu) <= GATE_INNER_HALF and abs(cv) <= GATE_INNER_HALF:
                events.extend(self._on_gate_passed(sim_time))

        self._prev_w = w
        self._prev_pos = np.asarray(drone_pos_w, dtype=float).copy()

        # When the target gate changes, w must be recomputed next step.
        return events

    def _on_gate_passed(self, sim_time):
        events = []
        gate = self.gates[self.active_gate_index]
        if self.active_gate_index == 0:
            self.race_start_time = sim_time
            events.append(f"[RACE] START  - crossed start gate at t={sim_time:.2f}s")
        race_t = sim_time - (self.race_start_time if self.race_start_time else sim_time)
        self.last_gate_race_time = race_t
        self.gate_split_times.append((gate.index, race_t))
        if self.active_gate_index > 0:
            events.append(
                f"[RACE] Gate {gate.index}/{len(self.gates) - 1} passed - "
                f"race time {race_t:.2f}s")
        if self.active_gate_index == len(self.gates) - 1:
            self.race_finish_time = sim_time
            events.append(
                f"[RACE] FINISH - course complete in {race_t:.2f}s")
        else:
            self.active_gate_index += 1
            self._prev_w = None
        return events

    # ------------------------------------------------------------------
    def status_payload_fields(self, sim_time):
        start_ms = -1 if self.race_start_time is None else int(self.race_start_time * 1000)
        finish_ns = -1 if self.race_finish_time is None else int(self.race_finish_time * 1e9)
        last_ms = -1 if self.last_gate_race_time is None else int(self.last_gate_race_time * 1000)
        return {
            "race_start_boot_time_ms": start_ms,
            "race_finish_time_ns": finish_ns,
            "active_gate_index": self.active_gate_index,
            "last_gate_race_time_ms": last_ms,
        }

    def check_collisions(self, sim_time, robot_node, drone_pos_w, min_interval=0.25):
        """Detect contacts via the supervisor API. Returns (collision_id,
        threat_level, impulse) or None. Ground contact before the race start
        (sitting on the pad) is ignored."""
        if sim_time - self._last_collision_time < min_interval:
            return None
        try:
            points = robot_node.getContactPoints(True)
        except Exception:
            return None
        if not points:
            return None
        if self.race_start_time is None and drone_pos_w[2] < 0.25:
            return None  # resting on the launch pad
        self._last_collision_time = sim_time

        collision_id = wire.COLLISION_ID_ENVIRONMENT
        # Near a gate frame -> classify as gate strike.
        p = np.asarray(drone_pos_w)
        for g in self.gates:
            u, v, w = g.local_coords(p)
            if abs(w) < 0.5 and abs(u) < 1.6 and abs(v) < 1.6:
                collision_id = wire.COLLISION_ID_GATE
                break
        return collision_id, 2, 1.0

    def timed_out(self, sim_time):
        if self.race_start_time is None:
            return False
        return (sim_time - self.race_start_time) > self.max_duration_s

    def summary(self):
        lines = ["[RACE] ===== RACE SUMMARY ====="]
        total = len(self.gates)
        passed = len(self.gate_split_times)
        lines.append(f"[RACE] Gates passed: {passed}/{total}")
        for idx, t in self.gate_split_times:
            lines.append(f"[RACE]   gate {idx:2d}  split {t:7.2f}s")
        if self.race_finish_time is not None and self.race_start_time is not None:
            lines.append(
                f"[RACE] FINAL TIME: {self.race_finish_time - self.race_start_time:.2f}s")
        else:
            lines.append("[RACE] Did not finish")
        return lines
