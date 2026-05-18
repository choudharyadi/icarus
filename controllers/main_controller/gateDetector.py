import cv2
import numpy as np
import pyvista as pv

class GatePnPDetector:
    def __init__(self):
        self.W, self.H = 324, 324
        self.f = (self.W / 2) / np.tan(0.87 / 2)
        
        self.camera_matrix = np.array([[self.f, 0, self.W/2], [0, self.f, self.H/2], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))

        # Define model points in a VERTICAL plane (YZ)
        # Normal vector is [1, 0, 0] (Forward)
        self.GATE_RADIUS = 0.5
        pts = 16
        angles = np.linspace(0, 2*np.pi, pts, endpoint=False)
        self.gate_3d_pts = np.array([[0, self.GATE_RADIUS*np.cos(a), self.GATE_RADIUS*np.sin(a)] for a in angles], dtype=np.float32)

    def get_relative_gate_data(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None: return []

        # Masking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])),
                              cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
                if len(approx) >= 16:
                    indices = np.linspace(0, len(approx)-1, 16, dtype=int)
                    img_pts = approx[indices].astype(np.float32)

                    success, rvec, tvec = cv2.solvePnP(self.gate_3d_pts, img_pts, self.camera_matrix, self.dist_coeffs)

                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        
                        # --- THE CORRECTED AXIS MAPPING ---
                        # 1. Translation: OpenCV (x,y,z) -> Drone (z, -x, -y)
                        t = tvec.flatten()
                        pos = np.array([t[2], -t[0], -t[1]])
                        
                        # 2. Rotation Matrix Re-alignment
                        # We need to rotate the OpenCV rotation matrix so that 
                        # the "Forward" direction aligns with the Drone's X-axis.
                        # This basis change fixes the "vertical" vs "horizontal" tilt.
                        R_basis = np.array([
                            [0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]
                        ])
                        rmat_final = R_basis @ rmat
                        
                        results.append((pos, rmat_final))
        return results

class PnPVisualizer:
    def __init__(self):
        self.gates = []

    def add_gate(self, pos, rmat):
        self.gates.append((pos, rmat))
        print(f"Gate detected at position: {pos}")
        print(f"Rotation matrix:\n{rmat}\n")

    def run(self):
        print(f"Total gates detected: {len(self.gates)}")

if __name__ == "__main__":
    detector = GatePnPDetector()
    viz = PnPVisualizer()
    path = '/Users/adityachoudhary/Documents/Coding/autonomous-parkour-drone/controllers/main_controller/pictures/camera_376.54.png'
    
    for pos, rmat in detector.get_relative_gate_data(path):
        viz.add_gate(pos, rmat)
    
    viz.run()