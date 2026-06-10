# import cv2
# import numpy as np
# import pyvista as pv

# class GatePnPDetector:
#     def __init__(self):
#         self.W, self.H = 324, 324
#         self.f = (self.W / 2) / np.tan(0.87 / 2)
        
#         self.camera_matrix = np.array([[self.f, 0, self.W/2], [0, self.f, self.H/2], [0, 0, 1]], dtype=np.float32)
#         self.dist_coeffs = np.zeros((4, 1))

#         # Define model points in a VERTICAL plane (YZ)
#         # Normal vector is [1, 0, 0] (Forward)
#         self.GATE_RADIUS = 0.5
#         pts = 16
#         angles = np.linspace(0, 2*np.pi, pts, endpoint=False)
#         self.gate_3d_pts = np.array([[0, self.GATE_RADIUS*np.cos(a), self.GATE_RADIUS*np.sin(a)] for a in angles], dtype=np.float32)

#     def get_relative_gate_data(self, image_path):
#         frame = cv2.imread(image_path)
#         if frame is None: return []

#         # Masking
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])),
#                               cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])))

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         results = []

#         for cnt in contours:
#             if cv2.contourArea(cnt) > 100:
#                 approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
#                 if len(approx) >= 16:
#                     indices = np.linspace(0, len(approx)-1, 16, dtype=int)
#                     img_pts = approx[indices].astype(np.float32)

#                     success, rvec, tvec = cv2.solvePnP(self.gate_3d_pts, img_pts, self.camera_matrix, self.dist_coeffs)

#                     if success:
#                         rmat, _ = cv2.Rodrigues(rvec)
                        
#                         # --- THE CORRECTED AXIS MAPPING ---
#                         # 1. Translation: OpenCV (x,y,z) -> Drone (z, -x, -y)
#                         t = tvec.flatten()
#                         pos = np.array([t[2], -t[0], -t[1]])
                        
#                         # 2. Rotation Matrix Re-alignment
#                         # We need to rotate the OpenCV rotation matrix so that 
#                         # the "Forward" direction aligns with the Drone's X-axis.
#                         # This basis change fixes the "vertical" vs "horizontal" tilt.
#                         R_basis = np.array([
#                             [0, 0, 1],
#                             [-1, 0, 0],
#                             [0, -1, 0]
#                         ])
#                         rmat_final = R_basis @ rmat
                        
#                         results.append((pos, rmat_final))
#         return results

# class PnPVisualizer:
#     def __init__(self):
#         self.gates = []

#     def add_gate(self, pos, rmat):
#         self.gates.append((pos, rmat))
#         print(f"Gate detected at position: {pos}")
#         print(f"Rotation matrix:\n{rmat}\n")

#     def run(self):
#         print(f"Total gates detected: {len(self.gates)}")

# if __name__ == "__main__":
#     detector = GatePnPDetector()
#     viz = PnPVisualizer()
#     path = '/Users/adityachoudhary/Documents/Coding/autonomous-parkour-drone/controllers/main_controller/pictures/camera_376.54.png'
    
#     for pos, rmat in detector.get_relative_gate_data(path):
#         viz.add_gate(pos, rmat)
    
#     viz.run()


import cv2
import numpy as np
import os

class GateDetector:
    def __init__(self, width=640, height=360, show_debug_window=None):
        self.W, self.H = width, height
        self.fx = 320.0
        self.fy = 320.0
        self.camera_pitch = np.radians(20.0)
        self.min_inner_area = max(80.0, self.W * self.H * 0.00035)
        self.max_reprojection_error = 6.0
        self.min_gate_distance = 0.5
        self.max_gate_distance = 35.0
        self.last_debug_frame = None
        if show_debug_window is None:
            show_debug_window = os.environ.get("ICARUS_DEBUG_WINDOW", "1") != "0"
        self.show_debug_window = show_debug_window
        
        # --- GATE DIMENSIONS ---
        self.GATE_INNER_W = 1.5  
        self.GATE_INNER_H = 1.5  

        # --- CAMERA INTRINSIC MATRIX ---
        self.camera_matrix = np.array([
            [self.fx,       0, self.W / 2],
            [      0, self.fy, self.H / 2],
            [      0,       0,          1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # 3D object points for PnP (inner ring corners in local gate frame)
        half_w = self.GATE_INNER_W / 2.0
        half_h = self.GATE_INNER_H / 2.0
        self.object_points = np.array([
            [-half_w,  half_h, 0.0],  # Top-Left
            [ half_w,  half_h, 0.0],  # Top-Right
            [ half_w, -half_h, 0.0],  # Bottom-Right
            [-half_w, -half_h, 0.0]   # Bottom-Left
        ], dtype=np.float32)

    def _sort_corners(self, pts):
        """
        Robustly sorts 4 corners into [TL, TR, BR, BL] order.
        Stable at any rotation unlike the sum/diff trick.
        """
        pts = pts[np.argsort(pts[:, 1])]  # sort all by Y

        # If middle two points are ambiguous in Y, re-sort by X first
        if abs(pts[1, 1] - pts[2, 1]) < 10:
            pts[1:3] = pts[1:3][np.argsort(pts[1:3, 0])]

        top = pts[:2][np.argsort(pts[:2, 0])]  # top two, left to right
        bot = pts[2:][np.argsort(pts[2:, 0])]  # bottom two, left to right

        tl, tr = top[0], top[1]
        bl, br = bot[0], bot[1]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _build_red_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_red = cv2.inRange(hsv, np.array([0, 100, 60]), np.array([12, 255, 255]))
        high_red = cv2.inRange(hsv, np.array([168, 100, 60]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(low_red, high_red)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _draw_debug_viz(self, frame, mask, contours, candidates, detections):
        """
        Multi-panel debug visualization showing every step of detection.
        
        Panel layout:
        ┌─────────────────┬─────────────────┐
        │   RAW FRAME     │   HSV RED MASK  │
        ├─────────────────┼─────────────────┤
        │ CONTOURS+CORNERS│  PnP RESULT     │
        └─────────────────┴─────────────────┘
        """
        H, W = frame.shape[:2]
        
        # ── Panel 1: annotated camera view ──────────────────────────────
        p1 = frame.copy()
        cv2.line(p1, (W // 2 - 12, H // 2), (W // 2 + 12, H // 2), (255, 255, 0), 1)
        cv2.line(p1, (W // 2, H // 2 - 12), (W // 2, H // 2 + 12), (255, 255, 0), 1)
        for candidate in candidates:
            color = (50, 220, 50) if candidate["accepted"] else (40, 40, 220)
            box = np.int32(candidate["image_points"])
            cv2.polylines(p1, [box], True, color, 2)
            x, y = box[0]
            label = candidate["label"]
            cv2.putText(p1, label, (int(x), max(18, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(p1, "CAMERA + ACCEPTED/REJECTED", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Panel 2: HSV red mask ───────────────────────────────────────
        p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # make 3-channel for stacking
        cv2.putText(p2, "CLEANED RED MASK", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # ── Panel 3: All contours + sorted corner labels ─────────────────
        p3 = frame.copy()
        corner_labels = ["TL", "TR", "BR", "BL"]
        corner_colors = [
            (255, 0,   0),   # TL → Blue
            (0,   255, 0),   # TR → Green
            (0,   0,   255), # BR → Red
            (0,   255, 255)  # BL → Yellow
        ]

        cv2.drawContours(p3, contours, -1, (70, 70, 70), 1)
        for candidate in candidates:
            for pt, label, color in zip(candidate["image_points"], corner_labels, corner_colors):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(p3, (x, y), 4, color, -1)
                cv2.putText(p3, label, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        cv2.putText(p3, "CORNERS (TL/TR/BR/BL)", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Panel 4: PnP result ──────────────────────────────────────────
        p4 = np.zeros_like(frame)  # black background
        cv2.putText(p4, "PnP POSE ESTIMATE", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        accepted = sum(candidate["accepted"] for candidate in candidates)
        lines = [
            f"accepted: {accepted} / candidates: {len(candidates)}",
            f"min area: {self.min_inner_area:.0f}px  max fit error: {self.max_reprojection_error:.1f}px",
            f"range: {self.min_gate_distance:.1f}-{self.max_gate_distance:.0f}m",
        ]
        for idx, candidate in enumerate(candidates[:7]):
            if candidate["accepted"]:
                pos = candidate["position"]
                lines.append(
                    f"{idx + 1}: F/L/U {pos[0]:+.1f} {pos[1]:+.1f} {pos[2]:+.1f}m "
                    f"fit {candidate['reprojection_error']:.1f}px"
                )
            else:
                lines.append(f"{idx + 1}: {candidate['label']}")
        for idx, line in enumerate(lines):
            color = (100, 255, 100) if idx == 0 and accepted else (220, 220, 220)
            cv2.putText(p4, line, (10, 48 + idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

        # ── Stitch 4 panels into 2x2 grid ───────────────────────────────
        top    = np.hstack([p1, p2])
        bottom = np.hstack([p3, p4])
        grid   = np.vstack([top, bottom])

        # Panel divider lines
        cv2.line(grid, (W, 0),  (W, H*2),  (60, 60, 60), 1)  # vertical
        cv2.line(grid, (0, H),  (W*2, H),  (60, 60, 60), 1)  # horizontal

        self.last_debug_frame = grid
        if self.show_debug_window:
            try:
                cv2.imshow("Gate Detection Debug", grid)
                cv2.waitKey(1)
            except Exception:
                self.show_debug_window = False

    def get_relative_gate_data_from_frame(self, frame):
        """
        Accepts a BGR frame directly (e.g. from Webots camera).
        Returns list of (rel_pos, R_matrix) for each detected gate.
        """
        if frame is None:
            print("Error: No frame provided")
            return []

        mask = self._build_red_mask(frame)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_data = []
        candidates = []

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if hierarchy[0][i][3] != -1 and area > self.min_inner_area:
                    parent_index = hierarchy[0][i][3]
                    parent_area = cv2.contourArea(contours[parent_index])
                    opening_ratio = area / parent_area if parent_area > 0 else 0.0
                    rect = cv2.minAreaRect(cnt)
                    rect_width, rect_height = rect[1]
                    if min(rect_width, rect_height) < 8:
                        continue

                    box_pts = cv2.boxPoints(rect)
                    image_points = self._sort_corners(box_pts)
                    success, rvec, tvec = cv2.solvePnP(
                        self.object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    candidate = {
                        "accepted": False,
                        "image_points": image_points,
                        "label": "rejected: PnP failed",
                    }

                    if success:
                        cam_x = tvec[0][0]
                        cam_y = tvec[1][0]
                        cam_z = tvec[2][0]
                        camera_pos = np.array([cam_z, -cam_x, -cam_y])
                        cos_pitch = np.cos(self.camera_pitch)
                        sin_pitch = np.sin(self.camera_pitch)
                        rel_pos = np.array([
                            cos_pitch * camera_pos[0] - sin_pitch * camera_pos[2],
                            camera_pos[1],
                            sin_pitch * camera_pos[0] + cos_pitch * camera_pos[2]
                        ])

                        R_matrix, _ = cv2.Rodrigues(rvec)
                        projected, _ = cv2.projectPoints(
                            self.object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
                        )
                        reprojection_error = float(np.mean(
                            np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
                        ))
                        distance = float(np.linalg.norm(rel_pos))
                        aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)
                        candidate["position"] = rel_pos
                        candidate["reprojection_error"] = reprojection_error

                        rejection = None
                        if reprojection_error > self.max_reprojection_error:
                            rejection = f"fit {reprojection_error:.1f}px"
                        elif not self.min_gate_distance <= distance <= self.max_gate_distance:
                            rejection = f"range {distance:.1f}m"
                        elif rel_pos[0] <= 0.2:
                            rejection = "behind camera"
                        elif aspect_ratio > 3.5:
                            rejection = f"shape {aspect_ratio:.1f}:1"
                        elif not 0.08 <= opening_ratio <= 0.75:
                            rejection = f"opening {opening_ratio:.2f}"

                        if rejection is None:
                            duplicate = any(np.linalg.norm(rel_pos - pos) < 0.75 for pos, _ in found_data)
                            if duplicate:
                                rejection = "duplicate"
                            else:
                                found_data.append((rel_pos, R_matrix))
                                candidate["accepted"] = True
                                candidate["label"] = (
                                    f"gate {distance:.1f}m fit {reprojection_error:.1f}px"
                                )

                        if rejection is not None:
                            candidate["label"] = f"rejected: {rejection}"

                    candidates.append(candidate)

        self._draw_debug_viz(frame, mask, contours, candidates, found_data)
        return found_data 
    
