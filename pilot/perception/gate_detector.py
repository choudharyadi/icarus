"""Square-gate detector: HSV red mask -> contour hierarchy -> inner-opening
quad -> PnP pose.

The gate's inner opening is a known 1.5 m x 1.5 m square (spec 3.7), and the
camera intrinsics are exactly specified (spec 3.8), so solvePnP on the four
inner corners yields metric range and bearing. Detections are filtered hard:
reprojection error, plausible range, aspect ratio and opening ratio, because
a single bad pose injected into map fusion costs more than a missed frame.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from frames import cam_to_body


@dataclass
class GateDetection:
    pos_cam: np.ndarray            # OpenCV camera frame [m]
    pos_body: np.ndarray           # body FRD [m]
    distance: float
    reprojection_error: float
    image_points: np.ndarray       # 4x2 (TL, TR, BR, BL)
    area: float


class GateDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        w, h = cfg.image_width, cfg.image_height
        self.min_inner_area = max(80.0, w * h * cfg.min_inner_area_frac)

        self.camera_matrix = np.array([
            [cfg.fx, 0, cfg.cx],
            [0, cfg.fy, cfg.cy],
            [0, 0, 1],
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        half = cfg.gate_inner_size / 2.0
        self.object_points = np.array([
            [-half, -half, 0.0],   # TL (camera frame: x right, y down)
            [half, -half, 0.0],    # TR
            [half, half, 0.0],     # BR
            [-half, half, 0.0],    # BL
        ], dtype=np.float32)

        self.last_annotated = None

    # ------------------------------------------------------------------
    @staticmethod
    def _build_red_mask(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low = cv2.inRange(hsv, np.array([0, 100, 60]),
                          np.array([12, 255, 255]))
        high = cv2.inRange(hsv, np.array([168, 100, 60]),
                           np.array([180, 255, 255]))
        mask = cv2.bitwise_or(low, high)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    @staticmethod
    def _sort_corners(pts):
        """Sort 4 points into TL, TR, BR, BL (stable at moderate rotations)."""
        pts = pts[np.argsort(pts[:, 1])]
        if abs(pts[1, 1] - pts[2, 1]) < 10:
            pts[1:3] = pts[1:3][np.argsort(pts[1:3, 0])]
        top = pts[:2][np.argsort(pts[:2, 0])]
        bot = pts[2:][np.argsort(pts[2:, 0])]
        return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

    # ------------------------------------------------------------------
    def detect(self, frame, annotate=False):
        """Returns list[GateDetection], strongest (largest) first."""
        cfg = self.cfg
        mask = self._build_red_mask(frame)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        annotated = frame.copy() if annotate else None

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                # Inner opening = child contour inside a red outer contour.
                if hierarchy[0][i][3] == -1 or area < self.min_inner_area:
                    continue
                parent_area = cv2.contourArea(contours[hierarchy[0][i][3]])
                opening_ratio = area / parent_area if parent_area > 0 else 0.0
                if not 0.08 <= opening_ratio <= 0.80:
                    continue

                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if min(rw, rh) < 8:
                    continue
                if max(rw, rh) / max(1e-6, min(rw, rh)) > cfg.max_aspect_ratio:
                    continue

                image_points = self._sort_corners(cv2.boxPoints(rect))
                ok, rvec, tvec = cv2.solvePnP(
                    self.object_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE)
                if not ok:
                    continue

                projected, _ = cv2.projectPoints(
                    self.object_points, rvec, tvec,
                    self.camera_matrix, self.dist_coeffs)
                reproj = float(np.mean(np.linalg.norm(
                    projected.reshape(-1, 2) - image_points, axis=1)))
                if reproj > cfg.max_reprojection_error:
                    continue

                pos_cam = tvec.flatten().astype(float)
                if pos_cam[2] <= 0.2:
                    continue
                distance = float(np.linalg.norm(pos_cam))
                if not cfg.min_distance <= distance <= cfg.max_distance:
                    continue

                # Duplicate suppression (same gate seen twice via mask noise).
                if any(np.linalg.norm(pos_cam - d.pos_cam) < 0.75
                       for d in detections):
                    continue

                detections.append(GateDetection(
                    pos_cam=pos_cam,
                    pos_body=cam_to_body(pos_cam),
                    distance=distance,
                    reprojection_error=reproj,
                    image_points=image_points,
                    area=area,
                ))

                if annotate:
                    cv2.polylines(annotated, [np.int32(image_points)], True,
                                  (60, 220, 60), 2)
                    tl = image_points[0]
                    cv2.putText(
                        annotated,
                        f"{distance:.1f}m e{reproj:.1f}px",
                        (int(tl[0]), max(15, int(tl[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 60), 1,
                        cv2.LINE_AA)

        detections.sort(key=lambda d: -d.area)
        if annotate:
            self.last_annotated = annotated
        return detections
