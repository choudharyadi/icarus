"""Per-run logging: CSV flight log, event log and a JSON summary."""

import csv
import json
import os
import time
from datetime import datetime


class RunLogger:
    def __init__(self, base_dir="runs"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base_dir, f"run_{stamp}")
        os.makedirs(self.dir, exist_ok=True)

        self._csv_file = open(os.path.join(self.dir, "flight_log.csv"), "w",
                              newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "t_wall", "t_boot_ms", "phase", "n", "e", "d",
            "vn", "ve", "vd", "roll", "pitch", "yaw",
            "active_gate", "detections", "frame_id"])
        self._events_path = os.path.join(self.dir, "events.log")
        self._summary = {"started": stamp}
        self._t0 = time.time()
        self._rows = 0

    def row(self, snap, phase, n_detections, frame_id):
        self._csv.writerow([
            f"{snap.t_wall - self._t0:.3f}", snap.t_boot_ms, phase,
            f"{snap.pos_ned[0]:.3f}", f"{snap.pos_ned[1]:.3f}",
            f"{snap.pos_ned[2]:.3f}",
            f"{snap.vel_ned[0]:.3f}", f"{snap.vel_ned[1]:.3f}",
            f"{snap.vel_ned[2]:.3f}",
            f"{snap.roll:.4f}", f"{snap.pitch:.4f}", f"{snap.yaw:.4f}",
            snap.active_gate_index, n_detections, frame_id])
        self._rows += 1
        if self._rows % 100 == 0:
            self._csv_file.flush()

    def event(self, msg):
        with open(self._events_path, "a") as f:
            f.write(f"{time.time() - self._t0:9.3f}  {msg}\n")

    def set_summary(self, **kwargs):
        self._summary.update(kwargs)
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(self._summary, f, indent=2)

    def close(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
