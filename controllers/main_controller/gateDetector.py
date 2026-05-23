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

class GateDetector:
    def __init__(self):
        self.W, self.H = 324, 324
        self.FOV = 0.87
        self.f = (self.W / 2) / np.tan(self.FOV / 2)
        self.cam_offset = np.array([0.03, 0.0, 0.01])
        
        # --- GATE DIMENSIONS ---
        self.GATE_INNER_W = 1.5  
        self.GATE_INNER_H = 1.5  

        # --- CAMERA INTRINSIC MATRIX ---
        self.camera_matrix = np.array([
            [self.f,    0, self.W / 2],
            [   0, self.f, self.H / 2],
            [   0,    0,            1]
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

    def _draw_debug_viz(self, frame, contours, hierarchy, detections):
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
        
        # ── Panel 1: Raw frame ──────────────────────────────────────────
        p1 = frame.copy()
        cv2.putText(p1, "RAW FRAME", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # ── Panel 2: HSV red mask ───────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, np.array([0,   120, 70]), np.array([10,  255, 255]))
        m2  = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # make 3-channel for stacking
        cv2.putText(p2, "HSV RED MASK", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # ── Panel 3: All contours + sorted corner labels ─────────────────
        p3 = frame.copy()
        corner_labels = ["TL", "TR", "BR", "BL"]
        corner_colors = [
            (255, 0,   0),   # TL → Blue
            (0,   255, 0),   # TR → Green
            (0,   0,   255), # BR → Red
            (0,   255, 255)  # BL → Yellow
        ]

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                
                if hierarchy[0][i][3] == -1:
                    # Outer contour → draw dim white
                    cv2.drawContours(p3, [cnt], 0, (80, 80, 80), 1)
                else:
                    # Inner contour → this is a gate candidate
                    if area > 80:
                        # Draw the minAreaRect box in orange
                        rect = cv2.minAreaRect(cnt)
                        box_pts = cv2.boxPoints(rect)
                        cv2.drawContours(p3, [np.int32(box_pts)], 0, (0, 165, 255), 2)

                        # Sort corners and label each one
                        sorted_pts = self._sort_corners(box_pts)
                        for j, (pt, label, color) in enumerate(
                                zip(sorted_pts, corner_labels, corner_colors)):
                            x, y = int(pt[0]), int(pt[1])
                            cv2.circle(p3, (x, y), 5, color, -1)
                            cv2.putText(p3, label, (x + 6, y - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                        # Show area
                        cx, cy = int(rect[0][0]), int(rect[0][1])
                        cv2.putText(p3, f"area:{int(area)}", (cx - 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    else:
                        # Inner contour but too small → draw dim red
                        cv2.drawContours(p3, [cnt], 0, (0, 0, 120), 1)

        cv2.putText(p3, "CORNERS (TL/TR/BR/BL)", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Panel 4: PnP result ──────────────────────────────────────────
        p4 = np.zeros_like(frame)  # black background
        cv2.putText(p4, "PnP POSE ESTIMATE", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if detections:
            for idx, (pos, R) in enumerate(detections):
                y_start = 45 + idx * 80
                cv2.putText(p4, f"Gate {idx+1}:", (10, y_start),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(p4, f"  X (fwd):  {pos[0]:+.2f} m", (10, y_start + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
                cv2.putText(p4, f"  Y (left): {pos[1]:+.2f} m", (10, y_start + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
                cv2.putText(p4, f"  Z (up):   {pos[2]:+.2f} m", (10, y_start + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

                # Draw a small top-down 2D map showing gate direction
                map_cx, map_cy = W - 60, y_start + 25
                cv2.circle(p4, (map_cx, map_cy), 4, (0, 255, 255), -1)  # drone dot
                scale = 8  # pixels per meter
                gate_px = int(map_cx + pos[1] * scale)  # Y = left/right
                gate_py = int(map_cy - pos[0] * scale)  # X = forward
                cv2.rectangle(p4,
                            (gate_px - 5, gate_py - 5),
                            (gate_px + 5, gate_py + 5),
                            (0, 165, 255), 1)
                cv2.line(p4, (map_cx, map_cy), (gate_px, gate_py), (80, 80, 80), 1)
        else:
            cv2.putText(p4, "No gates detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(p4, "Check HSV mask panel --^", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # ── Stitch 4 panels into 2x2 grid ───────────────────────────────
        top    = np.hstack([p1, p2])
        bottom = np.hstack([p3, p4])
        grid   = np.vstack([top, bottom])

        # Panel divider lines
        cv2.line(grid, (W, 0),  (W, H*2),  (60, 60, 60), 1)  # vertical
        cv2.line(grid, (0, H),  (W*2, H),  (60, 60, 60), 1)  # horizontal

        cv2.imshow("Gate Detection Debug", grid)
        cv2.waitKey(1)

    def get_relative_gate_data_from_frame(self, frame):
        """
        Accepts a BGR frame directly (e.g. from Webots camera).
        Returns list of (rel_pos, R_matrix) for each detected gate.
        """
        if frame is None:
            print("Error: No frame provided")
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect red (hue wraps around 0/180 in HSV so we need two ranges)
        m1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_data = []

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                # Only process inner contours (those with a parent) above minimum area
                if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > 80:
                    rect = cv2.minAreaRect(cnt)
                    box_pts = cv2.boxPoints(rect)

                    # Use robust corner sorting
                    image_points = self._sort_corners(box_pts)

                    # Solve PnP — recover 3D gate pose from 2D image points
                    success, rvec, tvec = cv2.solvePnP(
                        self.object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        cam_x = tvec[0][0]
                        cam_y = tvec[1][0]
                        cam_z = tvec[2][0]

                        # Remap from OpenCV camera frame (X right, Y down, Z forward)
                        # to drone frame (X forward, Y left, Z up)
                        rel_pos = np.array([
                            cam_z + self.cam_offset[0],
                            -cam_x + self.cam_offset[1],
                            -cam_y + self.cam_offset[2]
                        ])

                        R_matrix, _ = cv2.Rodrigues(rvec)
                        found_data.append((rel_pos, R_matrix))


        self._draw_debug_viz(frame, contours, hierarchy, found_data)
        return found_data 
    

