"""Thread-safe shared vehicle / race / track state.

Comms threads write, the control loop reads consistent snapshots.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackGate:
    gate_id: int
    pos_ned: np.ndarray          # (3,)
    quat_ned: np.ndarray         # (w, x, y, z)
    width: float
    height: float


@dataclass
class Snapshot:
    t_wall: float = 0.0
    t_boot_ms: int = 0

    # Vehicle
    pos_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rates: np.ndarray = field(default_factory=lambda: np.zeros(3))
    armed: bool = False

    have_position: bool = False
    have_attitude: bool = False
    last_telemetry_wall: float = -1.0

    # Race
    active_gate_index: int = 0
    race_started: bool = False
    race_finished: bool = False
    race_start_ms: int = -1
    race_finish_ns: int = -1
    last_gate_race_time_ms: int = -1
    sim_boot_time_ms: int = 0

    # Track
    gates: list = field(default_factory=list)
    collisions: int = 0
    last_collision_wall: float = -1.0


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._s = Snapshot()

        # Vision frames are large: store outside the snapshot.
        self._frame = None           # np BGR image
        self._frame_id = -1
        self._frame_sim_ns = 0
        self._frames_received = 0

    # ----------------------- telemetry writers -----------------------
    def update_attitude(self, roll, pitch, yaw, rates, t_boot_ms):
        with self._lock:
            s = self._s
            s.roll, s.pitch, s.yaw = roll, pitch, yaw
            s.rates = np.asarray(rates, dtype=float)
            s.t_boot_ms = t_boot_ms
            s.have_attitude = True
            s.last_telemetry_wall = time.time()

    def update_position(self, pos, vel, t_boot_ms):
        with self._lock:
            s = self._s
            s.pos_ned = np.asarray(pos, dtype=float)
            s.vel_ned = np.asarray(vel, dtype=float)
            s.t_boot_ms = t_boot_ms
            s.have_position = True
            s.last_telemetry_wall = time.time()

    def update_heartbeat(self, armed):
        with self._lock:
            self._s.armed = armed
            self._s.last_telemetry_wall = time.time()

    def update_race_status(self, sim_boot_ms, start_ms, finish_ns,
                           active_gate, last_gate_ms):
        with self._lock:
            s = self._s
            s.sim_boot_time_ms = sim_boot_ms
            s.race_start_ms = start_ms
            s.race_finish_ns = finish_ns
            s.active_gate_index = active_gate
            s.last_gate_race_time_ms = last_gate_ms
            s.race_started = start_ms >= 0
            s.race_finished = finish_ns > 0

    def set_track(self, gates):
        with self._lock:
            if not self._s.gates:
                self._s.gates = gates

    def record_collision(self):
        with self._lock:
            self._s.collisions += 1
            self._s.last_collision_wall = time.time()

    # ----------------------- vision -----------------------
    def push_frame(self, frame_id, img, sim_time_ns):
        with self._lock:
            self._frame = img
            self._frame_id = frame_id
            self._frame_sim_ns = sim_time_ns
            self._frames_received += 1

    def latest_frame(self, newer_than_id=-1):
        with self._lock:
            if self._frame is None or self._frame_id <= newer_than_id:
                return None, -1, 0
            return self._frame, self._frame_id, self._frame_sim_ns

    @property
    def frames_received(self):
        with self._lock:
            return self._frames_received

    # ----------------------- readers -----------------------
    def snapshot(self):
        with self._lock:
            s = self._s
            out = Snapshot(
                t_wall=time.time(),
                t_boot_ms=s.t_boot_ms,
                pos_ned=s.pos_ned.copy(),
                vel_ned=s.vel_ned.copy(),
                roll=s.roll, pitch=s.pitch, yaw=s.yaw,
                rates=s.rates.copy(),
                armed=s.armed,
                have_position=s.have_position,
                have_attitude=s.have_attitude,
                last_telemetry_wall=s.last_telemetry_wall,
                active_gate_index=s.active_gate_index,
                race_started=s.race_started,
                race_finished=s.race_finished,
                race_start_ms=s.race_start_ms,
                race_finish_ns=s.race_finish_ns,
                last_gate_race_time_ms=s.last_gate_race_time_ms,
                sim_boot_time_ms=s.sim_boot_time_ms,
                gates=list(s.gates),
                collisions=s.collisions,
                last_collision_wall=s.last_collision_wall,
            )
            return out
