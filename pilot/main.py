#!/usr/bin/env python3
"""Icarus race pilot - AI Grand Prix Virtual Qualifier entry point.

Connects to the simulator (real DCL sim or the Webots dcl_sim_bridge replica)
over MAVLink UDP 14550 + vision UDP 5600, then autonomously arms, takes off
and races the course.

Usage:
    python3 pilot/main.py [--viz]
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")  # spec mandates MAVLink 2

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comms.mavlink_io import MavlinkIO
from comms.vision_rx import VisionRX
from config import PilotConfig
from frames import body_to_ned
from guidance import Guidance
from logging_util import RunLogger
from perception.gate_detector import GateDetector
from planning.track import Track
from state import SharedState


def main():
    parser = argparse.ArgumentParser(description="Icarus race pilot")
    parser.add_argument("--viz", action="store_true",
                        help="show live detection window")
    parser.add_argument("--save-frames", action="store_true",
                        help="periodically save annotated frames to the run dir")
    args = parser.parse_args()

    cfg = PilotConfig()
    if args.viz:
        cfg.viz = True
    if args.save_frames:
        cfg.save_debug_frames = True

    print("=" * 60)
    print("ICARUS RACE PILOT - AI Grand Prix Virtual Qualifier")
    print("=" * 60)

    state = SharedState()
    logger = RunLogger(cfg.log_dir)
    mav = MavlinkIO(state, cfg.comms)
    vision = VisionRX(state, cfg.comms)
    detector = GateDetector(cfg.perception)
    track = Track(cfg.perception)
    guidance = Guidance(cfg, mav, track, logger)

    vision.start()
    mav.connect()

    period = 1.0 / cfg.race.control_hz
    last_frame_id = -1
    last_status_print = 0.0
    next_frame_save = 0.0
    ever_connected_wall = time.time()

    cv2 = None
    if cfg.viz:
        import cv2 as _cv2
        cv2 = _cv2

    try:
        while True:
            tick = time.time()
            snap = state.snapshot()

            # ---- perception ------------------------------------------
            detections = []
            frame, frame_id, _ = state.latest_frame(newer_than_id=last_frame_id)
            if frame is not None:
                last_frame_id = frame_id
                annotate = cfg.viz or (cfg.save_debug_frames and
                                       tick >= next_frame_save)
                dets = detector.detect(frame, annotate=annotate)
                for det in dets:
                    est_center = snap.pos_ned + body_to_ned(
                        det.pos_body, snap.roll, snap.pitch, snap.yaw)
                    detections.append((det, est_center))
                    track.apply_detection(est_center, det,
                                          snap.active_gate_index)

                if cfg.viz and detector.last_annotated is not None:
                    cv2.imshow("icarus pilot", detector.last_annotated)
                    cv2.waitKey(1)
                if cfg.save_debug_frames and tick >= next_frame_save and \
                        detector.last_annotated is not None:
                    import cv2 as _c
                    _c.imwrite(os.path.join(
                        logger.dir, f"frame_{frame_id:06d}.jpg"),
                        detector.last_annotated)
                    next_frame_save = tick + 2.0

            # ---- guidance --------------------------------------------
            guidance.update(snap, detections)
            logger.row(snap, guidance.phase, len(detections), last_frame_id)

            # ---- console status --------------------------------------
            if tick - last_status_print > 2.0:
                last_status_print = tick
                speed = float(np.linalg.norm(snap.vel_ned))
                print(f"[{guidance.phase:11s}] "
                      f"N{snap.pos_ned[0]:+7.2f} E{snap.pos_ned[1]:+7.2f} "
                      f"D{snap.pos_ned[2]:+6.2f} | {speed:4.2f} m/s | "
                      f"gate {snap.active_gate_index} | "
                      f"frames {state.frames_received} | "
                      f"det {len(detections)}", flush=True)

            # ---- link watchdog ----------------------------------------
            if snap.last_telemetry_wall > 0:
                ever_connected_wall = snap.last_telemetry_wall
            if time.time() - ever_connected_wall > cfg.comms.link_loss_exit_s:
                print("Link lost - exiting", flush=True)
                break

            # ---- pace loop --------------------------------------------
            elapsed = time.time() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Interrupted", flush=True)
    finally:
        mav.stop()
        vision.stop()
        logger.close()
        print(f"Logs written to {logger.dir}", flush=True)


if __name__ == "__main__":
    main()
