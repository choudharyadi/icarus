"""Onboard stabilized flight controller for the simulated quadrotor.

This emulates the DCL simulator's built-in stabilized controller: contestant
software streams SET_POSITION_TARGET_LOCAL_NED / SET_ATTITUDE_TARGET setpoints
and this cascade turns them into motor commands.

Cascade layout (all in Webots-native frames; NED conversion happens at the
MAVLink boundary in `dcl_sim_bridge.py`):

    position error -> velocity target -> attitude target -> motor mixing

The attitude/velocity gains are derived from the proven Bitcraze Crazyflie
Webots controller and tuned for faster racing flight.
"""

import math

import numpy as np

from frames import wrap_pi


class Setpoint:
    """Most recent control target, already converted into Webots frames."""

    MODE_NONE = "none"
    MODE_VELOCITY = "velocity"      # vel_w (3,) world frame [m/s]
    MODE_POSITION = "position"      # pos_w (3,) world frame [m]
    MODE_RATES_THRUST = "rates"     # body rates [rad/s] + thrust 0..1

    def __init__(self):
        self.mode = self.MODE_NONE
        self.vel_w = np.zeros(3)
        self.pos_w = np.zeros(3)
        self.yaw_w = None           # absolute yaw target (rad) or None
        self.yaw_rate_w = None      # yaw rate target (rad/s) or None
        self.body_rates = np.zeros(3)
        self.thrust = 0.0
        self.stamp = -1.0           # sim time when received


class FlightController:
    # Velocity / acceleration envelope (m/s, m/s^2)
    MAX_H_VEL = 3.5
    MAX_V_VEL = 2.0
    MAX_H_ACC = 3.0

    # Position loop
    KP_POS = 1.2

    # Velocity -> attitude loop
    KP_VEL = 1.2
    KD_VEL = 0.25
    VEL_ERR_CLIP = 1.0
    TILT_LIMIT = 0.35  # rad - cap commanded roll/pitch so the attitude loop
                       # can always track and the vehicle never departs

    # Attitude loop
    KP_ATT_RP = 0.5
    KD_ATT_RP = 0.1
    KP_YAW_RATE = 1.0
    KP_YAW_ANGLE = 2.2
    MAX_YAW_RATE = 2.0

    # Altitude loop
    KP_Z = 10.0
    KI_Z = 5.0
    KD_Z = 5.0
    ALT_INT_CLIP = 2.0
    HOVER_FF = 48.0

    MOTOR_MAX = 600.0

    # Setpoint staleness: hold position if the client stops streaming.
    SETPOINT_TIMEOUT = 0.6

    def __init__(self):
        self.armed = False
        self.setpoint = Setpoint()

        self._alt_target = None
        self._hold_pos_w = None
        self._alt_integrator = 0.0
        self._past_alt_error = 0.0
        self._past_vx_error = 0.0
        self._past_vy_error = 0.0
        self._past_pitch_error = 0.0
        self._past_roll_error = 0.0
        self._prev_vel_cmd_body = np.zeros(2)

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    def reset(self):
        self.__init__()

    # ------------------------------------------------------------------
    def update(self, dt, sim_time, pos_w, vel_w, rpy_w, gyro_flu):
        """One control step.  Returns motor speeds [m1, m2, m3, m4] (signed
        velocities are applied by the caller). All inputs are Webots-native.
        """
        if not self.armed or dt <= 0.0:
            return [0.0, 0.0, 0.0, 0.0]

        roll, pitch, yaw = rpy_w
        sp = self.setpoint

        stale = sp.mode == Setpoint.MODE_NONE or (
            sp.stamp >= 0.0 and (sim_time - sp.stamp) > self.SETPOINT_TIMEOUT
        )

        if sp.mode == Setpoint.MODE_RATES_THRUST and not stale:
            return self._rates_thrust_step(sp, gyro_flu)

        # ---- resolve a velocity command in world frame -----------------
        if stale:
            # Hold the position captured when the stream went silent.
            if self._hold_pos_w is None:
                self._hold_pos_w = pos_w.copy()
            vel_cmd_w = self.KP_POS * (self._hold_pos_w - pos_w)
            vel_cmd_w[2] = 0.0
            alt_target = self._hold_pos_w[2]
            yaw_rate_cmd = 0.0
        else:
            self._hold_pos_w = None
            if sp.mode == Setpoint.MODE_POSITION:
                err = sp.pos_w - pos_w
                vel_cmd_w = self.KP_POS * err
                vel_cmd_w[2] = 0.0
                alt_target = sp.pos_w[2]
            else:  # MODE_VELOCITY
                vel_cmd_w = sp.vel_w.copy()
                vz = np.clip(vel_cmd_w[2], -self.MAX_V_VEL, self.MAX_V_VEL)
                vel_cmd_w[2] = 0.0
                if self._alt_target is None:
                    self._alt_target = pos_w[2]
                # Integrate climb-rate command into an altitude target the
                # tight altitude PID can track; anti-windup keeps it near
                # the actual altitude.
                self._alt_target += vz * dt
                self._alt_target = float(np.clip(
                    self._alt_target, pos_w[2] - 0.4, pos_w[2] + 0.4))
                self._alt_target = float(np.clip(self._alt_target, 0.1, 30.0))
                alt_target = self._alt_target

            # ---- yaw -----------------------------------------------------
            if sp.yaw_rate_w is not None:
                yaw_rate_cmd = sp.yaw_rate_w
            elif sp.yaw_w is not None:
                yaw_rate_cmd = self.KP_YAW_ANGLE * wrap_pi(sp.yaw_w - yaw)
            else:
                yaw_rate_cmd = 0.0

        if sp.mode == Setpoint.MODE_POSITION and not stale:
            self._alt_target = alt_target
        yaw_rate_cmd = float(np.clip(yaw_rate_cmd, -self.MAX_YAW_RATE, self.MAX_YAW_RATE))

        # ---- limit horizontal speed ------------------------------------
        h_speed = math.hypot(vel_cmd_w[0], vel_cmd_w[1])
        if h_speed > self.MAX_H_VEL:
            vel_cmd_w[:2] *= self.MAX_H_VEL / h_speed

        # ---- world -> body-FLU horizontal velocity ----------------------
        cy, sy = math.cos(yaw), math.sin(yaw)
        fwd_cmd = vel_cmd_w[0] * cy + vel_cmd_w[1] * sy
        side_cmd = -vel_cmd_w[0] * sy + vel_cmd_w[1] * cy

        # Slew-limit the body velocity command (smooth accelerations).
        cmd = np.array([fwd_cmd, side_cmd])
        delta = cmd - self._prev_vel_cmd_body
        max_step = self.MAX_H_ACC * dt
        norm = np.linalg.norm(delta)
        if norm > max_step:
            cmd = self._prev_vel_cmd_body + delta * (max_step / norm)
        self._prev_vel_cmd_body = cmd
        fwd_cmd, side_cmd = float(cmd[0]), float(cmd[1])

        # ---- measured body velocity -------------------------------------
        v_fwd = vel_w[0] * cy + vel_w[1] * sy
        v_side = -vel_w[0] * sy + vel_w[1] * cy

        return self._velocity_attitude_step(
            dt, fwd_cmd, side_cmd, yaw_rate_cmd, alt_target,
            roll, pitch, gyro_flu[2], pos_w[2], v_fwd, v_side)

    # ------------------------------------------------------------------
    def _velocity_attitude_step(self, dt, des_vx, des_vy, des_yaw_rate,
                                des_alt, roll, pitch, yaw_rate,
                                alt, v_x, v_y):
        # Velocity PID -> attitude targets
        vx_error = des_vx - v_x
        vx_deriv = (vx_error - self._past_vx_error) / dt
        vy_error = des_vy - v_y
        vy_deriv = (vy_error - self._past_vy_error) / dt
        desired_pitch = self.KP_VEL * np.clip(vx_error, -self.VEL_ERR_CLIP, self.VEL_ERR_CLIP) \
            + self.KD_VEL * vx_deriv
        desired_roll = -self.KP_VEL * np.clip(vy_error, -self.VEL_ERR_CLIP, self.VEL_ERR_CLIP) \
            - self.KD_VEL * vy_deriv
        desired_pitch = float(np.clip(desired_pitch, -self.TILT_LIMIT, self.TILT_LIMIT))
        desired_roll = float(np.clip(desired_roll, -self.TILT_LIMIT, self.TILT_LIMIT))
        self._past_vx_error = vx_error
        self._past_vy_error = vy_error

        # Altitude PID
        alt_error = des_alt - alt
        alt_deriv = (alt_error - self._past_alt_error) / dt
        self._alt_integrator += alt_error * dt
        self._alt_integrator = float(np.clip(
            self._alt_integrator, -self.ALT_INT_CLIP, self.ALT_INT_CLIP))
        alt_command = (self.KP_Z * alt_error + self.KD_Z * alt_deriv +
                       self.KI_Z * self._alt_integrator + self.HOVER_FF)
        self._past_alt_error = alt_error

        # Attitude PID
        pitch_error = desired_pitch - pitch
        pitch_deriv = (pitch_error - self._past_pitch_error) / dt
        roll_error = desired_roll - roll
        roll_deriv = (roll_error - self._past_roll_error) / dt
        yaw_rate_error = des_yaw_rate - yaw_rate
        roll_command = self.KP_ATT_RP * np.clip(roll_error, -1, 1) + self.KD_ATT_RP * roll_deriv
        pitch_command = -self.KP_ATT_RP * np.clip(pitch_error, -1, 1) - self.KD_ATT_RP * pitch_deriv
        yaw_command = self.KP_YAW_RATE * np.clip(yaw_rate_error, -1, 1)
        self._past_pitch_error = pitch_error
        self._past_roll_error = roll_error

        return self._mix(alt_command, roll_command, pitch_command, yaw_command)

    def _rates_thrust_step(self, sp, gyro_flu):
        """SET_ATTITUDE_TARGET body-rate + collective-thrust mode.

        Body rates arrive in FRD (MAVLink); gyro is FLU, so q/r flip sign.
        """
        alt_command = float(np.clip(sp.thrust, 0.0, 1.0)) * 2.0 * self.HOVER_FF
        p_err = sp.body_rates[0] - gyro_flu[0]
        q_err = sp.body_rates[1] - (-gyro_flu[1])
        r_err = sp.body_rates[2] - (-gyro_flu[2])
        roll_command = 0.3 * np.clip(p_err, -2, 2)
        pitch_command = -0.3 * np.clip(q_err, -2, 2)
        yaw_command = -1.0 * np.clip(r_err, -2, 2)
        return self._mix(alt_command, roll_command, pitch_command, yaw_command)

    def _mix(self, alt_command, roll_command, pitch_command, yaw_command):
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command
        return [float(np.clip(m, 0.0, self.MOTOR_MAX)) for m in (m1, m2, m3, m4)]
