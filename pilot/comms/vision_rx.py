"""FPV vision stream receiver.

Reassembles chunked JPEG frames from the simulator (UDP, spec section 4.6)
and pushes decoded BGR images into shared state. Incomplete/stale frames are
dropped aggressively - the control loop only ever wants the newest image.
"""

import socket
import struct
import threading

import cv2
import numpy as np

HEADER_FMT = "<IHHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAX_PENDING_FRAMES = 6


class VisionRX:
    def __init__(self, state, cfg):
        self.state = state
        self.cfg = cfg
        self._running = False
        self._thread = None
        self._sock = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.cfg.vision_listen_ip,
                         self.cfg.vision_listen_port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"Vision listener on udp {self.cfg.vision_listen_ip}:"
              f"{self.cfg.vision_listen_port}", flush=True)

    def stop(self):
        self._running = False

    def _loop(self):
        frames = {}
        while self._running:
            try:
                packet, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(packet) < HEADER_SIZE:
                continue

            (frame_id, chunk_id, total_chunks, jpeg_size, payload_size,
             sim_time_ns) = struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
            payload = packet[HEADER_SIZE:HEADER_SIZE + payload_size]

            entry = frames.setdefault(frame_id, {
                "chunks": {}, "total": total_chunks,
                "size": jpeg_size, "time": sim_time_ns})
            entry["chunks"][chunk_id] = payload

            if len(entry["chunks"]) == entry["total"]:
                data = b"".join(entry["chunks"].get(i, b"")
                                for i in range(entry["total"]))
                del frames[frame_id]
                if len(data) != entry["size"]:
                    continue
                img = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is not None:
                    self.state.push_frame(frame_id, img, entry["time"])

            # Drop stale partial frames so memory stays bounded.
            if len(frames) > MAX_PENDING_FRAMES:
                for fid in sorted(frames)[:-MAX_PENDING_FRAMES]:
                    del frames[fid]
