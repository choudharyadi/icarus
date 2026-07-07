"""Spawns the contestant pilot as a separate OS process, exactly as it runs
against the real simulator, and pipes its console output into the Webots
console with a [PILOT] prefix.

Disable with environment variable ICARUS_AUTOPILOT=0 (e.g. to run the pilot
manually from a terminal, or against a remote machine).
"""

import atexit
import os
import subprocess
import sys
import threading


class PilotLauncher:
    def __init__(self, project_root, extra_args=None):
        self.project_root = project_root
        self.proc = None
        self.extra_args = extra_args or []

    @property
    def enabled(self):
        return os.environ.get("ICARUS_AUTOPILOT", "1") != "0"

    def start(self):
        if not self.enabled:
            print("[BRIDGE] ICARUS_AUTOPILOT=0 - start pilot manually: "
                  "python3 pilot/main.py")
            return
        main_py = os.path.join(self.project_root, "pilot", "main.py")
        if not os.path.exists(main_py):
            print(f"[BRIDGE] Pilot entry not found at {main_py}")
            return
        cmd = [sys.executable, "-u", main_py] + self.extra_args
        env = dict(os.environ)
        env.setdefault("ICARUS_VIZ", "1")
        self.proc = subprocess.Popen(
            cmd, cwd=self.project_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._pump_output, daemon=True).start()
        atexit.register(self.stop)
        print(f"[BRIDGE] Pilot launched (pid {self.proc.pid})")

    def _pump_output(self):
        try:
            for line in self.proc.stdout:
                print(f"[PILOT] {line.rstrip()}", flush=True)
        except Exception:
            pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
