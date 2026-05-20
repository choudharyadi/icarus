import cv2
import numpy as np
import pyvista as pv

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

        # 3D Object Points for PnP (Inner ring corners in local frame)
        half_w = self.GATE_INNER_W / 2.0
        half_h = self.GATE_INNER_H / 2.0
        self.object_points = np.array([
            [-half_w,  half_h, 0.0],  # Top-Left
            [ half_w,  half_h, 0.0],  # Top-Right
            [ half_w, -half_h, 0.0],  # Bottom-Right
            [-half_w, -half_h, 0.0]   # Bottom-Left
        ], dtype=np.float32)

    def get_relative_gate_data(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None: 
            print(f"Error: Could not look up image at {image_path}")
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_data = []

        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > 80:
                    rect = cv2.minAreaRect(cnt)
                    box_pts = cv2.boxPoints(rect)
                    
                    # Sort corners (TL, TR, BR, BL)
                    pts_sum = box_pts.sum(axis=1)
                    pts_diff = np.diff(box_pts, axis=1).flatten()
                    tl = box_pts[np.argmin(pts_sum)]
                    br = box_pts[np.argmax(pts_sum)]
                    tr = box_pts[np.argmin(pts_diff)]
                    bl = box_pts[np.argmax(pts_diff)]
                    
                    image_points = np.array([tl, tr, br, bl], dtype=np.float32)

                    # Solve PnP
                    success, rvec, tvec = cv2.solvePnP(
                        self.object_points, 
                        image_points, 
                        self.camera_matrix, 
                        self.dist_coeffs, 
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        cam_x, cam_y, cam_z = tvec[0][0], tvec[1][0], tvec[2][0]

                        # Map to drone coordinate system (X=Forward, Y=Left, Z=Up)
                        rel_pos = np.array([
                            cam_z + self.cam_offset[0],
                            -cam_x + self.cam_offset[1],
                            -cam_y + self.cam_offset[2]
                        ])

                        # Get rotation matrix matrix from rvec
                        R_matrix, _ = cv2.Rodrigues(rvec)
                        
                        found_data.append((rel_pos, R_matrix))

                        # Draw green tracking lines on 2D image
                        green_box = np.int32(box_pts)
                        cv2.drawContours(frame, [green_box], 0, (0, 255, 0), 2)
                        cv2.putText(frame, f"PnP Dist: {rel_pos[0]:.2f}m", (int(tl[0]), int(tl[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Show OpenCV window briefly, closes automatically when PyVista launches
        cv2.imshow("Gate Detection - PnP Solver Baseline", frame)
        cv2.waitKey(1500) 
        cv2.destroyAllWindows()
        
        return found_data
class RelativeVisualizer:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.plotter.set_background("black")
        self.plotter.show_axes()
        self.plotter.add_title("Drone FPV Camera Perspective", font_size=10)
        
        # --- BUILD RECONSTRUCTED DRONE BODY MESH ---
        drone_center = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.08, height=0.04)
        
        arm_thickness = 0.015
        arm1 = pv.Box(bounds=(-0.25, 0.25, -arm_thickness, arm_thickness, -arm_thickness/2, arm_thickness/2))
        arm2 = pv.Box(bounds=(-arm_thickness, arm_thickness, -0.25, 0.25, -arm_thickness/2, arm_thickness/2))
        
        arm1.rotate_z(45, inplace=True)
        arm2.rotate_z(45, inplace=True)
        
        drone_mesh = drone_center.merge(arm1).merge(arm2)
        
        # rendering drone configuration at the origin space
        self.plotter.add_mesh(drone_mesh, color="cyan", opacity=0.9, show_edges=True, label="Drone (Origin)")
        
        # Add local forward-heading arrow indicator (Drone pointing straight down +X Axis)
        self.plotter.add_arrows(cent=np.array([0, 0, 0]), direction=np.array([1, 0, 0]), mag=0.3, color="blue")

    def build_3d_gate_mesh(self, outer_dim=2.7, inner_dim=1.5, depth=0.26):
        """Generates a clean 3D hollow square frame mesh via boolean extraction"""
        outer_box = pv.Box(bounds=(
            -depth/2, depth/2, 
            -outer_dim/2, outer_dim/2, 
            -outer_dim/2, outer_dim/2
        ))
        inner_box = pv.Box(bounds=(
            -depth, depth,
            -inner_dim/2, inner_dim/2, 
            -inner_dim/2, inner_dim/2
        ))
        
        outer_tri = outer_box.triangulate()
        inner_tri = inner_box.triangulate()
        
        gate_mesh = outer_tri - inner_tri
        return gate_mesh

    def add_relative_gate(self, rel_pos, R_matrix):
        gate_mesh = self.build_3d_gate_mesh()

        R_drone_conv = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        R_final = R_drone_conv @ R_matrix @ R_drone_conv.T

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_final
        transform_matrix[:3, 3] = rel_pos

        gate_mesh.transform(transform_matrix, inplace=True)

        self.plotter.add_mesh(gate_mesh, color="orange", opacity=0.8, show_edges=True, label="Detected Gate")
        
        normal_vector = R_final @ np.array([1, 0, 0])
        self.plotter.add_arrows(cent=rel_pos, direction=normal_vector, mag=0.6, color="red")

        print(f"Rendered 3D Gate at: X={rel_pos[0]:.2f}m, Y={rel_pos[1]:.2f}m, Z={rel_pos[2]:.2f}m")

    def run(self):
        self.plotter.add_legend()
        
        # --- LOCK CAMERA TO DRONE FIRST PERSON PERSPECTIVE (FPV) ---
        # position: Place the camera at the drone's origin (0,0,0)
        # focal_point: Look straight down the +X axis (Forward)
        # view_up: Keep the +Z axis pointing up (Drone Sky direction)
        self.plotter.camera.position = (0.0, 0.0, 0.0)
        self.plotter.camera.focal_point = (1.0, 0.0, 0.0)
        self.plotter.camera.view_up = (0.0, 0.0, 1.0)
        
        # Match the virtual camera FOV roughly to your physical camera setup (OpenCV FOV is ~50 degrees)
        self.plotter.camera.view_angle = 50.0 

        self.plotter.show()

if __name__ == "__main__":
    detector = GateDetector()
    viz = RelativeVisualizer()

    img_path = '/Users/adityachoudhary/Documents/Coding/autonomous-parkour-drone/controllers/main_controller/pictures/camera_1040.58.png'
    results = detector.get_relative_gate_data(img_path)

    if len(results) == 0:
        print("No gates detected in image. Launching empty 3D environment layout...")
        
    for pos, R_mat in results:
        viz.add_relative_gate(pos, R_mat)

    viz.run()