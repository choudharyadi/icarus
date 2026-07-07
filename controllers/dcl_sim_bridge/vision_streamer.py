"""FPV camera stream emulation.

Encodes the Webots camera image as JPEG and ships it over UDP using the exact
chunked packet format from the AI Grand Prix spec (section 4.6):

    header  "<IHHIIQ"  frame_id, chunk_id, total_chunks, jpeg_size,
                       payload_size, sim_time_ns
    payload jpeg slice
"""

import socket

import cv2
import numpy as np

import wire


class VisionStreamer:
    def __init__(self, camera, client_ip="127.0.0.1", client_port=5600,
                 jpeg_quality=80):
        self.camera = camera
        self.addr = (client_ip, client_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.jpeg_quality = jpeg_quality
        self.frame_id = 0
        self.width = camera.getWidth()
        self.height = camera.getHeight()

    def send_frame(self, sim_time):
        raw = self.camera.getImage()
        if raw is None:
            return
        img = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 4))
        bgr = np.ascontiguousarray(img[:, :, :3])  # Webots delivers BGRA; drop alpha
        ok, jpeg = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return

        data = jpeg.tobytes()
        sim_time_ns = int(sim_time * 1e9)
        total = max(1, (len(data) + wire.VISION_CHUNK_PAYLOAD - 1)
                    // wire.VISION_CHUNK_PAYLOAD)
        for chunk_id in range(total):
            off = chunk_id * wire.VISION_CHUNK_PAYLOAD
            chunk = data[off:off + wire.VISION_CHUNK_PAYLOAD]
            header = wire.pack_vision_header(
                self.frame_id, chunk_id, total, len(data), len(chunk),
                sim_time_ns)
            try:
                self.sock.sendto(header + chunk, self.addr)
            except OSError:
                return
        self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
