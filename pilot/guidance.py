"""Flight phase state machine and setpoint generation.

Phases:
    INIT     wait for telemetry and the track description (the simulator
             transmits gate poses per spec - see mavlink_io track parsing)
    ARM      request arming
    TAKEOFF  climb to first-gate altitude, aim at first gate
    RACE     carrot-follow the racing line through all gates
    FINISH   hold position, report

Outputs SET_POSITION_TARGET_LOCAL_NED velocity+yaw setpoints at the control
rate; velocity magnitude is scheduled by the racing line.
"""

import time

import numpy as np

from frames import bearing_to, wrap_pi
from planning.racing_line import RacingLine


class Guidance:
    INIT = "INIT"
    ARM = "ARM"
    TAKEOFF = "TAKEOFF"
    RACE = "RACE"
    FINISH = "FINISH"

    def __init__(self, cfg, mav, track, logger):
        self.cfg = cfg
        self.rc = cfg.race
        self.mav = mav
        self.track = track
        self.logger = logger

        self.phase = self.INIT
        self.line = None
        self._armed_request_t = -10.0
        self._last_active_gate = 0
        self._finish_logged = False
        self._hold_pos = None
        self._hold_yaw = 0.0

    # ------------------------------------------------------------------
    def _set_phase(self, phase, why=""):
        if phase != self.phase:
            msg = f"PHASE {self.phase} -> {phase}" + (f" ({why})" if why else "")
            print(msg, flush=True)
            self.logger.event(msg)
            self.phase = phase

    # ------------------------------------------------------------------
    def update(self, snap, detections):
        """One control tick. snap: state.Snapshot; detections: list of
        (GateDetection, est_center_ned)."""
        # Safety: stale telemetry -> command zero velocity, no decisions.
        if snap.last_telemetry_wall > 0 and \
                time.time() - snap.last_telemetry_wall > self.cfg.comms.telemetry_timeout_s:
            self.mav.send_velocity_yaw(np.zeros(3), snap.yaw)
            return

        if self.phase == self.INIT:
            self._update_init(snap)
        elif self.phase == self.ARM:
            self._update_arm(snap)
        elif self.phase == self.TAKEOFF:
            self._update_takeoff(snap)
        elif self.phase == self.RACE:
            self._update_race(snap, detections)
        elif self.phase == self.FINISH:
            self._update_finish(snap)

    # ------------------------------------------------------------------
    def _update_init(self, snap):
        if not (snap.have_position and snap.have_attitude):
            return
        if snap.gates:
            self.track.load_from_track_data(snap.gates)
            self._set_phase(self.ARM, "track received")

    def _update_arm(self, snap):
        if snap.armed:
            self._set_phase(self.TAKEOFF)
            return
        if time.time() - self._armed_request_t > 1.0:
            self.mav.arm()
            self._armed_request_t = time.time()
        self.mav.send_velocity_yaw(np.zeros(3), snap.yaw)

    def _update_takeoff(self, snap):
        g0 = self.track.gates[0]
        target_d = float(g0.center[2])
        target_yaw = bearing_to(snap.pos_ned, g0.center)

        err_d = target_d - snap.pos_ned[2]      # NED down
        vz = float(np.clip(err_d * 1.2, -self.rc.climb_rate, self.rc.climb_rate))
        self.mav.send_velocity_yaw(np.array([0.0, 0.0, vz]), target_yaw)

        if abs(err_d) < self.rc.takeoff_alt_tolerance and \
                abs(snap.vel_ned[2]) < 0.2 and \
                abs(wrap_pi(target_yaw - snap.yaw)) < self.rc.takeoff_yaw_tolerance:
            self.line = RacingLine(self.track, self.rc)
            self.line.begin(snap.pos_ned)
            self._set_phase(self.RACE, "takeoff complete")

    # ------------------------------------------------------------------
    def _update_race(self, snap, detections):
        if snap.race_finished:
            self._enter_finish(snap)
            return

        # Race-status gate advancement clamps plan progress.
        if snap.active_gate_index != self._last_active_gate:
            self.logger.event(
                f"gate advance -> active {snap.active_gate_index} "
                f"(race {snap.last_gate_race_time_ms / 1000.0:.2f}s)")
            print(f"Gate {snap.active_gate_index - 1} passed "
                  f"(split {snap.last_gate_race_time_ms / 1000.0:.2f}s)",
                  flush=True)
            self._last_active_gate = snap.active_gate_index
            self.line.notify_gate_passed(snap.active_gate_index)

        cmd = self.line.command(snap.pos_ned)
        self._fly_towards(snap, cmd.carrot, cmd.speed, cmd.look_point)

        if cmd.finished:
            self._enter_finish(snap)

    def _fly_towards(self, snap, carrot, speed, look_point):
        d = carrot - snap.pos_ned
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            v_cmd = np.zeros(3)
        else:
            v_cmd = d / dist * speed
        v_cmd[2] = float(np.clip(v_cmd[2], -self.rc.v_vertical_max,
                                 self.rc.v_vertical_max))

        yaw_cmd = bearing_to(snap.pos_ned, look_point)
        # If the nose is far off target, slow down so the camera (and the
        # velocity controller) stay coordinated through turns.
        nose_err = abs(wrap_pi(yaw_cmd - snap.yaw))
        if nose_err > self.rc.yaw_rate_full_speed:
            scale = max(0.25, 1.0 - 0.6 * (nose_err - self.rc.yaw_rate_full_speed))
            v_cmd[:2] *= scale

        self.mav.send_velocity_yaw(v_cmd, yaw_cmd)

    # ------------------------------------------------------------------
    def _enter_finish(self, snap):
        if self.phase != self.FINISH:
            self._hold_pos = snap.pos_ned.copy()
            self._hold_yaw = snap.yaw
            self._set_phase(self.FINISH)

    def _update_finish(self, snap):
        if not self._finish_logged and snap.race_finished:
            t = (snap.race_finish_ns / 1e9) - (snap.race_start_ms / 1000.0)
            msg = f"RACE FINISHED - official time {t:.2f}s, " \
                  f"collisions {snap.collisions}"
            print(msg, flush=True)
            self.logger.event(msg)
            self.logger.set_summary(final_time_s=t, collisions=snap.collisions)
            self._finish_logged = True
        self.mav.send_position_yaw(self._hold_pos, self._hold_yaw)
