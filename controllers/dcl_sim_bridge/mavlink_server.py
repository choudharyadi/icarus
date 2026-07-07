"""MAVLink UDP server side of the DCL simulator bridge.

Mirrors the real AI Grand Prix simulator transport: the simulator *sends* to
the contestant client which listens on UDP 14550 (the official example client
uses `udpin:127.0.0.1:14550`), and processes whatever the client streams back
(heartbeats, timesync requests, setpoints, commands).

All methods are non-blocking and intended to be called from the Webots step
loop, so telemetry timing is locked to simulation time.
"""

import math

import numpy as np
from pymavlink import mavutil

import wire
from flight_controller import Setpoint
from frames import vec_ned_to_w, yaw_ned_to_w

POSITION_TARGET_TYPEMASK_X_IGNORE = mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
POSITION_TARGET_TYPEMASK_VX_IGNORE = mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
POSITION_TARGET_TYPEMASK_YAW_IGNORE = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE


class MavlinkServer:
    def __init__(self, client_ip="127.0.0.1", client_port=14550):
        self.conn = mavutil.mavlink_connection(
            f"udpout:{client_ip}:{client_port}",
            source_system=1, source_component=1)
        self.client_seen = False
        self.client_last_heard = -1.0

        # Callbacks wired up by the bridge
        self.on_arm = None            # fn(bool armed)
        self.on_reset = None          # fn()
        self.on_setpoint = None       # fn(Setpoint)
        self.get_pose = None          # fn() -> (pos_w (3,), yaw_w)

        self._track_transfer_id = 1
        self._last_track_send = -1e9

        # Telemetry schedule (sim seconds)
        self._next = {
            "heartbeat": 0.0,
            "attitude": 0.0,
            "lpn": 0.0,
            "imu": 0.0,
            "odometry": 0.0,
            "actuators": 0.0,
            "race_status": 0.0,
        }
        self._period = {
            "heartbeat": 0.5,
            "attitude": 0.02,
            "lpn": 0.02,
            "imu": 0.008,
            "odometry": 0.04,
            "actuators": 0.1,
            "race_status": 0.1,
        }

    # ------------------------------------------------------------------
    # RX
    # ------------------------------------------------------------------
    def process_incoming(self, sim_time, origin_w):
        """Drain the UDP socket; returns nothing. origin_w is the NED origin
        expressed in Webots world coordinates (for position setpoints)."""
        while True:
            try:
                msg = self.conn.recv_match(blocking=False)
            except (ConnectionResetError, OSError):
                return
            if msg is None:
                return
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            if mtype == "HEARTBEAT":
                self.client_seen = True
                self.client_last_heard = sim_time
            elif mtype == "TIMESYNC":
                # Client request: tc1 = client clock, ts1 = 0.
                if msg.ts1 == 0:
                    self.conn.mav.timesync_send(
                        int(sim_time * 1e9), msg.tc1)
            elif mtype == "COMMAND_LONG":
                self._handle_command(msg)
            elif mtype == "SET_POSITION_TARGET_LOCAL_NED":
                self._handle_position_target(msg, sim_time, origin_w)
            elif mtype == "SET_ATTITUDE_TARGET":
                self._handle_attitude_target(msg, sim_time)
            elif mtype == "SET_ACTUATOR_CONTROL_TARGET":
                # Raw motor control is accepted by the real sim; expose the
                # same surface but it is not used by our pilot.
                pass

    def _handle_command(self, msg):
        cmd = msg.command
        if cmd == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            armed = msg.param1 >= 0.5
            if self.on_arm:
                self.on_arm(armed)
            self.conn.mav.command_ack_send(cmd, mavutil.mavlink.MAV_RESULT_ACCEPTED)
        elif cmd == wire.MAVLINK_CMD_SIM_RESET:
            if self.on_reset:
                self.on_reset()
            self.conn.mav.command_ack_send(cmd, mavutil.mavlink.MAV_RESULT_ACCEPTED)

    def _handle_position_target(self, msg, sim_time, origin_w):
        sp = Setpoint()
        sp.stamp = sim_time
        mask = msg.type_mask
        frame = msg.coordinate_frame

        use_pos = not (mask & POSITION_TARGET_TYPEMASK_X_IGNORE)
        use_vel = not (mask & POSITION_TARGET_TYPEMASK_VX_IGNORE)
        use_yaw = not (mask & POSITION_TARGET_TYPEMASK_YAW_IGNORE)
        use_yaw_rate = not (mask & POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

        body_frame = frame in (mavutil.mavlink.MAV_FRAME_BODY_NED,
                               mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED)
        pos_w_now, yaw_w_now = self.get_pose() if self.get_pose else (np.zeros(3), 0.0)
        cy, sy = math.cos(yaw_w_now), math.sin(yaw_w_now)

        def body_frd_to_world(v_frd):
            # FRD body -> FLU body -> rotate by current yaw into world frame.
            v_flu = np.array([v_frd[0], -v_frd[1], -v_frd[2]])
            return np.array([
                v_flu[0] * cy - v_flu[1] * sy,
                v_flu[0] * sy + v_flu[1] * cy,
                v_flu[2],
            ])

        if use_pos:
            sp.mode = Setpoint.MODE_POSITION
            p_ned = np.array([msg.x, msg.y, msg.z])
            if body_frame:
                sp.pos_w = pos_w_now + body_frd_to_world(p_ned)
            else:
                sp.pos_w = origin_w + vec_ned_to_w(p_ned)
        elif use_vel:
            sp.mode = Setpoint.MODE_VELOCITY
            v_ned = np.array([msg.vx, msg.vy, msg.vz])
            if body_frame:
                sp.vel_w = body_frd_to_world(v_ned)
            else:
                sp.vel_w = vec_ned_to_w(v_ned)
        else:
            return

        if use_yaw_rate and not math.isnan(msg.yaw_rate):
            sp.yaw_rate_w = -msg.yaw_rate
        elif use_yaw and not math.isnan(msg.yaw):
            sp.yaw_w = yaw_ned_to_w(msg.yaw)

        if self.on_setpoint:
            self.on_setpoint(sp)

    def _handle_attitude_target(self, msg, sim_time):
        sp = Setpoint()
        sp.stamp = sim_time
        sp.mode = Setpoint.MODE_RATES_THRUST
        sp.body_rates = np.array([msg.body_roll_rate, msg.body_pitch_rate,
                                  msg.body_yaw_rate])
        sp.thrust = msg.thrust
        if self.on_setpoint:
            self.on_setpoint(sp)

    # ------------------------------------------------------------------
    # TX
    # ------------------------------------------------------------------
    def send_telemetry(self, sim_time, telem, armed, race_status):
        """telem: dict with NED-frame state produced by the bridge."""
        try:
            self._send_telemetry(sim_time, telem, armed, race_status)
        except OSError:
            pass  # client gone; UDP unreachable must never stop the sim

    def _send_telemetry(self, sim_time, telem, armed, race_status):
        ms = int(sim_time * 1000)
        us = int(sim_time * 1e6)

        if self._due("heartbeat", sim_time):
            base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            if armed:
                base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            self.conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                base_mode, 0, mavutil.mavlink.MAV_STATE_ACTIVE)

        if self._due("attitude", sim_time):
            r, p, y = telem["rpy_ned"]
            rs, ps, ys = telem["rates_frd"]
            self.conn.mav.attitude_send(ms, r, p, y, rs, ps, ys)

        if self._due("lpn", sim_time):
            pn = telem["pos_ned"]
            vn = telem["vel_ned"]
            self.conn.mav.local_position_ned_send(
                ms, pn[0], pn[1], pn[2], vn[0], vn[1], vn[2])

        if self._due("imu", sim_time):
            acc = telem["acc_frd"]
            gyr = telem["rates_frd"]
            self.conn.mav.highres_imu_send(
                us, acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2],
                0, 0, 0, 0, 0, 0, 0, 0xFFF, id=0)

        if self._due("odometry", sim_time):
            pn = telem["pos_ned"]
            vb = telem["vel_frd"]
            q = telem["quat_ned"]  # (w, x, y, z)
            rs, ps, ys = telem["rates_frd"]
            self.conn.mav.odometry_send(
                us,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mavutil.mavlink.MAV_FRAME_BODY_FRD,
                pn[0], pn[1], pn[2],
                [q[0], q[1], q[2], q[3]],
                vb[0], vb[1], vb[2],
                rs, ps, ys,
                [float('nan')] * 21, [float('nan')] * 21,
                reset_counter=telem.get("reset_counter", 0),
                estimator_type=mavutil.mavlink.MAV_ESTIMATOR_TYPE_NAIVE)

        if self._due("actuators", sim_time):
            motors = telem.get("motors", [0, 0, 0, 0])
            act = list(motors) + [0.0] * 28
            self.conn.mav.actuator_output_status_send(us, 4, act)

        if self._due("race_status", sim_time) and race_status is not None:
            payload = wire.pack_race_status(
                ms,
                race_status["race_start_boot_time_ms"],
                race_status["race_finish_time_ns"],
                race_status["active_gate_index"],
                race_status["last_gate_race_time_ms"])
            self.conn.mav.encapsulated_data_send(0, list(payload))

    def _due(self, key, sim_time):
        if sim_time >= self._next[key]:
            self._next[key] = sim_time + self._period[key]
            return True
        return False

    # ------------------------------------------------------------------
    def send_collision(self, collision_id, threat_level, impulse):
        # src=0 (MAV_COLLISION_SRC_ADSB placeholder), action=report-only.
        # `horizontal_minimum_delta` carries the impulse magnitude [kg m/s],
        # matching the field use documented in the official example client.
        try:
            self.conn.mav.collision_send(
                0, collision_id,
                mavutil.mavlink.MAV_COLLISION_ACTION_REPORT,
                int(threat_level), 0.0, 0.0, float(impulse))
        except OSError:
            pass

    def maybe_send_track(self, sim_time, gates, period=5.0):
        """Announce + transmit the track description (chunked), repeated
        periodically so late-joining clients always receive it."""
        if not self.client_seen or not gates:
            return
        if sim_time - self._last_track_send < period:
            return
        self._last_track_send = sim_time

        payload = wire.pack_track_payload(gates)
        transfer_id = self._track_transfer_id
        self._track_transfer_id = (self._track_transfer_id + 1) % 65535 or 1
        chunks = wire.chunk_track_payload(payload, transfer_id)

        # DATA_TRANSMISSION_HANDSHAKE repurposed as 'track data incoming':
        # width = transfer id, packets = chunk count.
        try:
            self.conn.mav.data_transmission_handshake_send(
                0, len(payload), transfer_id, 0, len(chunks), 0, 0)
            for seq, chunk in enumerate(chunks):
                self.conn.mav.encapsulated_data_send(seq, list(chunk))
        except OSError:
            pass
