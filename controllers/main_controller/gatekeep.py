import cv2
import numpy as np
import pyvista as pv

class GateDetector:
    def __init__(self):
        self.W, self.H = 324, 324
        self.FOV = 0.87
        self.GATE_RADIUS = 0.5 
        self.f = (self.W / 2) / np.tan(self.FOV / 2)
        self.cam_offset = np.array([0.03, 0.0, 0.01])

    def get_relative_gate_data(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None: return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_data = []

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > 100:
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (cx, cy), (MA, ma), angle_deg = ellipse
                        
                        # 1. TILT/ROTATION MATH
                        tilt = np.arccos(np.clip(ma / MA, 0, 1))
                        phi = np.radians(angle_deg)
                        
                        # Normal vector: where the gate 'faces'
                        nx = np.cos(tilt)
                        ny = np.sin(tilt) * np.cos(phi)
                        nz = np.sin(tilt) * np.sin(phi)
                        rel_normal = np.array([nx, ny, nz])

                        # 2. POSITION MATH
                        dist_z = ((self.GATE_RADIUS * 2) * self.f) / MA
                        rel_x_cam = (cx - self.W/2) * dist_z / self.f
                        rel_y_cam = (cy - self.H/2) * dist_z / self.f
                        
                        rel_pos = np.array([
                            dist_z + self.cam_offset[0],
                            -rel_x_cam + self.cam_offset[1],
                            -rel_y_cam + self.cam_offset[2]
                        ])
                        
                        found_data.append((rel_pos, rel_normal))
                        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

        # Removed cv2.imshow() for headless operation
        return found_data

class RelativeVisualizer:
    def __init__(self):
        self.gates = []
        
    def get_rotation_matrix(self, vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-6: return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    def add_relative_gate(self, rel_pos, rel_normal):
        num_pts = 60
        theta = np.linspace(0, 2*np.pi, num_pts)
        # Gate circle starts on YZ plane (facing X)
        circle_pts = np.column_stack((np.zeros_like(theta), 0.5*np.cos(theta), 0.5*np.sin(theta)))
        
        # Calculate alignment matrix
        R_align = self.get_rotation_matrix(np.array([1, 0, 0]), rel_normal)
        
        self.gates.append((rel_pos, rel_normal, R_align))
        print(f"Gate placed at relative offset: {rel_pos}")

    def run(self):
        print(f"Total gates processed: {len(self.gates)}")

if __name__ == "__main__":
    detector = GateDetector()
    viz = RelativeVisualizer()

    img_path = '/Users/adityachoudhary/Documents/Coding/autonomous-parkour-drone/controllers/main_controller/pictures/camera_349.12.png'
    results = detector.get_relative_gate_data(img_path)

    for pos, norm in results:
        viz.add_relative_gate(pos, norm)

    viz.run()