"""MAVLink client: telemetry RX thread, housekeeping TX thread and control
TX helpers.

Connection model matches the official PyAIPilotExample: the client listens on
UDP 14550 (`udpin`), the simulator transmits to it, and replies flow back to
the simulator's source address.
"""

import struct
import threading
import time

import numpy as np
from pymavlink import mavutil

from state import TrackGate

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2
MAVLINK_CMD_SIM_RESET = 31000

RACE_STATUS_FMT = "<BQqqIq"
TRACK_GATE_FMT = "<Hfffffffff"
TRACK_GATE_SIZE = struct.calcsize(TRACK_GATE_FMT)

VELOCITY_YAW_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

POSITION_YAW_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


class MavlinkIO:
    def __init__(self, state, cfg):
        self.state = state
        self.cfg = cfg
        self.conn = None
        self.boot_wall = time.time()
        self._running = False
        self._rx_thread = None
        self._housekeeping_thread = None

        self._track_chunks = {}
        self._track_expected = {}

    # ------------------------------------------------------------------
    def connect(self):
        self.conn = mavutil.mavlink_connection(
            "udpin:%s:%s" % (self.cfg.mavlink_listen_ip,
                             self.cfg.mavlink_listen_port),
            source_system=200, source_component=1)
        print("Waiting for simulator heartbeat on udp "
              f"{self.cfg.mavlink_listen_ip}:{self.cfg.mavlink_listen_port} ...",
              flush=True)
        self.conn.wait_heartbeat()
        print(f"Connected to simulator (system {self.conn.target_system})",
              flush=True)

        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self._housekeeping_thread = threading.Thread(
            target=self._housekeeping_loop, daemon=True)
        self._housekeeping_thread.start()

    def stop(self):
        self._running = False

    def _boot_ms(self):
        return int((time.time() - self.boot_wall) * 1000)

    # ------------------------------------------------------------------
    # RX
    # ------------------------------------------------------------------
    def _rx_loop(self):
        while self._running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except (ConnectionResetError, OSError):
                time.sleep(0.05)
                continue
            if msg is None:
                time.sleep(0.0005)
                continue
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            handler = getattr(self, "_on_" + mtype.lower(), None)
            if handler is not None:
                handler(msg)

    def _on_heartbeat(self, msg):
        armed = bool(msg.base_mode &
                     mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        self.state.update_heartbeat(armed)

    def _on_attitude(self, msg):
        self.state.update_attitude(
            msg.roll, msg.pitch, msg.yaw,
            (msg.rollspeed, msg.pitchspeed, msg.yawspeed),
            msg.time_boot_ms)

    def _on_local_position_ned(self, msg):
        self.state.update_position(
            (msg.x, msg.y, msg.z), (msg.vx, msg.vy, msg.vz),
            msg.time_boot_ms)

    def _on_odometry(self, msg):
        # LOCAL_POSITION_NED already feeds position; ODOMETRY kept for parity
        # with the real sim (body-frame velocity, quaternion, reset counter).
        pass

    def _on_highres_imu(self, msg):
        pass

    def _on_timesync(self, msg):
        pass

    def _on_collision(self, msg):
        self.state.record_collision()
        kind = "GATE" if msg.id == 1001 else "ENVIRONMENT"
        print(f"!! COLLISION ({kind}) impulse={msg.horizontal_minimum_delta:.2f}",
              flush=True)

    def _on_data_transmission_handshake(self, msg):
        transfer_id = msg.width
        self._track_chunks[transfer_id] = {}
        self._track_expected[transfer_id] = msg.packets

    def _on_encapsulated_data(self, msg):
        raw = bytes(msg.data)
        if not raw:
            return
        data_type = raw[0]
        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self._parse_race_status(raw)
        elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self._parse_track_chunk(msg, raw)

    def _parse_race_status(self, raw):
        (_, sim_boot_ms, start_ms, finish_ns, active_gate,
         last_gate_ms) = struct.unpack_from(RACE_STATUS_FMT, raw)
        self.state.update_race_status(sim_boot_ms, start_ms, finish_ns,
                                      active_gate, last_gate_ms)

    def _parse_track_chunk(self, msg, raw):
        _, transfer_id = struct.unpack_from("<BH", raw)
        if transfer_id not in self._track_expected:
            return
        self._track_chunks[transfer_id][msg.seqnr] = raw[3:]
        if len(self._track_chunks[transfer_id]) != self._track_expected[transfer_id]:
            return
        payload = b"".join(
            self._track_chunks[transfer_id][i]
            for i in range(self._track_expected[transfer_id]))
        del self._track_chunks[transfer_id]
        del self._track_expected[transfer_id]
        self._parse_track(payload)

    def _parse_track(self, payload):
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for _ in range(num_gates):
            (gate_id, px, py, pz, qw, qx, qy, qz, width,
             height) = struct.unpack_from(TRACK_GATE_FMT, payload)
            payload = payload[TRACK_GATE_SIZE:]
            gates.append(TrackGate(
                gate_id=gate_id,
                pos_ned=np.array([px, py, pz]),
                quat_ned=np.array([qw, qx, qy, qz]),
                width=width, height=height))
        gates.sort(key=lambda g: g.gate_id)
        self.state.set_track(gates)

    # ------------------------------------------------------------------
    # TX
    # ------------------------------------------------------------------
    def _housekeeping_loop(self):
        """Client obligations: heartbeat (>= 2 Hz) and timesync requests."""
        hb_period = 1.0 / self.cfg.heartbeat_hz
        ts_period = 1.0 / self.cfg.timesync_hz
        next_hb = 0.0
        next_ts = 0.0
        while self._running:
            now = time.time()
            try:
                if now >= next_hb:
                    self.conn.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    next_hb = now + hb_period
                if now >= next_ts:
                    self.conn.mav.timesync_send(int(time.time_ns()), 0)
                    next_ts = now + ts_period
            except OSError:
                pass
            time.sleep(0.02)

    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0)

    def disarm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0, 0, 0, 0, 0, 0, 0)

    def send_sim_reset(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0)

    def send_velocity_yaw(self, v_ned, yaw):
        """Velocity + absolute yaw setpoint in MAV_FRAME_LOCAL_NED."""
        self.conn.mav.set_position_target_local_ned_send(
            self._boot_ms(),
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_YAW_MASK,
            0, 0, 0,
            float(v_ned[0]), float(v_ned[1]), float(v_ned[2]),
            0, 0, 0,
            float(yaw), 0)

    def send_position_yaw(self, p_ned, yaw):
        self.conn.mav.set_position_target_local_ned_send(
            self._boot_ms(),
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            POSITION_YAW_MASK,
            float(p_ned[0]), float(p_ned[1]), float(p_ned[2]),
            0, 0, 0,
            0, 0, 0,
            float(yaw), 0)
