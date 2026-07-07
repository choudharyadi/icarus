"""Central configuration for the Icarus race pilot.

Every tunable lives here so race-day adjustments touch a single file.
Values are tuned for the Webots replica of the AI Grand Prix simulator and
are expected to transfer to the real sim with minor retuning of the speed
envelope only.
"""

import os
from dataclasses import dataclass, field


@dataclass
class CommsConfig:
    mavlink_listen_ip: str = "0.0.0.0"
    mavlink_listen_port: int = 14550
    vision_listen_ip: str = "0.0.0.0"
    vision_listen_port: int = 5600
    heartbeat_hz: float = 4.0
    timesync_hz: float = 5.0
    telemetry_timeout_s: float = 0.6      # setpoint safety: brake if stale
    link_loss_exit_s: float = 12.0        # orphan protection: exit pilot


@dataclass
class RaceConfig:
    control_hz: float = 50.0

    # Takeoff
    climb_rate: float = 1.2               # m/s
    takeoff_alt_tolerance: float = 0.18   # m
    takeoff_yaw_tolerance: float = 0.20   # rad

    # Speed envelope
    v_cruise: float = 3.4                 # straight-line speed cap, m/s
    v_gate: float = 2.0                   # speed while threading a gate, m/s
    v_min: float = 0.7                    # never plan slower than this, m/s
    decel: float = 2.0                    # planning deceleration, m/s^2
    v_vertical_max: float = 1.4           # climb/descend cap, m/s

    # Path geometry
    approach_dist: float = 1.7            # waypoint ahead of each gate, m
    exit_dist: float = 1.3                # waypoint beyond each gate, m
    lookahead: float = 1.6                # base carrot distance, m
    lookahead_speed_gain: float = 0.55    # carrot grows with planned speed
    lookahead_max: float = 2.4
    finish_brake_dist: float = 2.5        # slow-down zone after last gate

    # Yaw policy
    yaw_hold_dist: float = 1.3            # closer than this: stop re-aiming
    yaw_rate_full_speed: float = 0.9      # if |yaw err| above this, slow down

    # Turn-speed shaping
    turn_speed_k: float = 1.0             # higher = brake harder for turns


@dataclass
class PerceptionConfig:
    # Camera intrinsics (spec 3.8)
    image_width: int = 640
    image_height: int = 360
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 180.0

    gate_inner_size: float = 1.5          # m (spec 3.7)

    min_inner_area_frac: float = 0.00035
    max_reprojection_error: float = 6.0
    min_distance: float = 0.6
    max_distance: float = 30.0
    max_aspect_ratio: float = 3.5

    # Map fusion
    correction_enabled: bool = True
    correction_alpha: float = 0.25        # EMA blend per accepted detection
    correction_cap: float = 0.8           # max offset applied to a gate, m
    correction_min_dist: float = 1.5      # ignore detections closer than this
    correction_max_dist: float = 12.0
    association_radius: float = 3.0       # match detection -> track gate, m


@dataclass
class PilotConfig:
    comms: CommsConfig = field(default_factory=CommsConfig)
    race: RaceConfig = field(default_factory=RaceConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    log_dir: str = "runs"
    viz: bool = os.environ.get("ICARUS_VIZ", "0") == "1"
    save_debug_frames: bool = os.environ.get("ICARUS_SAVE_FRAMES", "0") == "1"
