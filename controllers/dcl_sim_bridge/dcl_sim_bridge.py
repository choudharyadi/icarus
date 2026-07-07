"""DCL simulator bridge - Webots controller.

Recreates the AI Grand Prix simulator interface (VADR-TS-002) on top of a
Webots world so the contestant pilot can be developed and raced locally
against the exact transport it will use in the competition:

  * MAVLink 2 over UDP -> client listening on 127.0.0.1:14550
    (HEARTBEAT, ATTITUDE, LOCAL_POSITION_NED, HIGHRES_IMU, ODOMETRY,
     ACTUATOR_OUTPUT_STATUS, TIMESYNC, COLLISION, ENCAPSULATED_DATA race
     status + track info)
  * Chunked JPEG FPV stream over UDP -> 127.0.0.1:5600 at ~30 Hz, 640x360
  * SET_POSITION_TARGET_LOCAL_NED / SET_ATTITUDE_TARGET control input
  * ARM/DISARM and SIM_RESET (31000) commands

Environment variables:
  ICARUS_AUTOPILOT=0   don't auto-launch pilot/main.py
  ICARUS_AUTOQUIT=1    quit Webots when the race finishes (CI/batch runs)
  ICARUS_CLIENT_IP     pilot host (default 127.0.0.1)
"""

import os
import sys

os.environ.setdefault("MAVLINK20", "1")  # spec mandates MAVLink 2

import numpy as np

from controller import Supervisor  # noqa: webots API

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flight_controller import FlightController
from frames import (body_flu_to_frd, quat_xyzw_to_rotmat, rotmat_to_euler_ned,
                    rotmat_to_quat_wxyz, rotmat_w_to_ned, vec_w_to_ned)
from mavlink_server import MavlinkServer
from pilot_launcher import PilotLauncher
from race_manager import RaceManager
from vision_streamer import VisionStreamer

CAMERA_PERIOD_STEPS = 4  # 4 x 8 ms = 32 ms ~ 30 Hz


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # ---- devices ------------------------------------------------------
    motors = [robot.getDevice(f"m{i}_motor") for i in range(1, 5)]
    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    accel = robot.getDevice("accelerometer")
    if accel is not None:
        accel.enable(timestep)

    camera = robot.getDevice("qualifier_camera")
    camera.enable(timestep * CAMERA_PERIOD_STEPS)

    # Settle one step so sensors return data.
    if robot.step(timestep) == -1:
        return

    origin_w = np.array(gps.getValues(), dtype=float)
    origin_w[2] = 0.0  # NED origin on the ground below the arming point

    self_node = robot.getSelf()
    start_translation = list(self_node.getField("translation").getSFVec3f())
    start_rotation = list(self_node.getField("rotation").getSFRotation())

    # ---- subsystems ----------------------------------------------------
    client_ip = os.environ.get("ICARUS_CLIENT_IP", "127.0.0.1")
    fc = FlightController()
    server = MavlinkServer(client_ip=client_ip)
    vision = VisionStreamer(camera, client_ip=client_ip)
    race = RaceManager(robot, origin_w)

    autoquit = os.environ.get("ICARUS_AUTOQUIT", "0") == "1"

    def on_arm(armed):
        if armed:
            fc.arm()
            print("[BRIDGE] Drone ARMED")
        else:
            fc.disarm()
            print("[BRIDGE] Drone DISARMED")

    def do_reset():
        print("[BRIDGE] SIM RESET requested")
        fc.reset()
        race.reset()
        self_node.getField("translation").setSFVec3f(start_translation)
        self_node.getField("rotation").setSFRotation(start_rotation)
        self_node.resetPhysics()

    server.on_arm = on_arm
    server.on_reset = do_reset
    server.on_setpoint = lambda sp: setattr(fc, "setpoint", sp)
    server.get_pose = lambda: (np.array(gps.getValues(), dtype=float),
                               imu.getRollPitchYaw()[2])

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    launcher = PilotLauncher(project_root)
    launcher.start()

    print("[BRIDGE] DCL sim bridge running - MAVLink -> "
          f"{client_ip}:14550, vision -> {client_ip}:5600")

    # ---- main loop ------------------------------------------------------
    step_count = 0
    last_time = robot.getTime()
    finish_announced_at = None
    motor_cmds = [0.0, 0.0, 0.0, 0.0]

    while robot.step(timestep) != -1:
        sim_time = robot.getTime()
        dt = sim_time - last_time
        last_time = sim_time
        step_count += 1

        # ---- sensors --------------------------------------------------
        pos_w = np.array(gps.getValues(), dtype=float)
        vel_w = np.array(gps.getSpeedVector(), dtype=float)
        rpy_w = imu.getRollPitchYaw()
        quat_w = imu.getQuaternion()
        gyro_flu = np.array(gyro.getValues(), dtype=float)
        if accel is not None:
            acc_flu = np.array(accel.getValues(), dtype=float)
        else:
            acc_flu = np.array([0.0, 0.0, 9.81])

        if not np.all(np.isfinite(vel_w)):
            vel_w = np.zeros(3)

        # ---- control input / flight controller -------------------------
        server.process_incoming(sim_time, origin_w)
        motor_cmds = fc.update(dt, sim_time, pos_w, vel_w, rpy_w, gyro_flu)
        motors[0].setVelocity(-motor_cmds[0])
        motors[1].setVelocity(motor_cmds[1])
        motors[2].setVelocity(-motor_cmds[2])
        motors[3].setVelocity(motor_cmds[3])

        # ---- telemetry --------------------------------------------------
        r_w = quat_xyzw_to_rotmat(quat_w)
        r_ned = rotmat_w_to_ned(r_w)
        pos_ned = vec_w_to_ned(pos_w - origin_w)
        vel_ned = vec_w_to_ned(vel_w)
        telem = {
            "rpy_ned": rotmat_to_euler_ned(r_ned),
            "rates_frd": body_flu_to_frd(gyro_flu),
            "pos_ned": pos_ned,
            "vel_ned": vel_ned,
            "vel_frd": r_ned.T @ vel_ned,
            "quat_ned": rotmat_to_quat_wxyz(r_ned),
            "acc_frd": body_flu_to_frd(acc_flu),
            "motors": motor_cmds,
        }
        server.send_telemetry(sim_time, telem, fc.armed,
                              race.status_payload_fields(sim_time))
        server.maybe_send_track(sim_time, race.track_records())

        # ---- race management --------------------------------------------
        for event in race.update(sim_time, pos_w):
            print(event, flush=True)
        hit = race.check_collisions(sim_time, self_node, pos_w)
        if hit is not None:
            cid, threat, impulse = hit
            kind = "GATE" if cid == 1001 else "ENVIRONMENT"
            print(f"[RACE] COLLISION with {kind} at t={sim_time:.2f}s")
            server.send_collision(cid, threat, impulse)

        # ---- vision ------------------------------------------------------
        if step_count % CAMERA_PERIOD_STEPS == 0:
            vision.send_frame(sim_time)

        # ---- end-of-run handling -----------------------------------------
        if race.race_finish_time is not None and finish_announced_at is None:
            finish_announced_at = sim_time
            for line in race.summary():
                print(line, flush=True)

        if autoquit:
            done = (finish_announced_at is not None
                    and sim_time - finish_announced_at > 3.0)
            expired = sim_time > float(os.environ.get("ICARUS_MAX_SIM_T", "480"))
            if done or expired or race.timed_out(sim_time):
                if finish_announced_at is None:
                    for line in race.summary():
                        print(line, flush=True)
                print("[BRIDGE] Auto-quit", flush=True)
                launcher.stop()
                robot.simulationQuit(0)
                break

    launcher.stop()


if __name__ == "__main__":
    main()
