import cv2
import numpy as np

class GateDetector:
    def __init__(self):
        self.W, self.H = 324, 324
        self.FOV = 0.87
        self.GATE_RADIUS = 0.5 
        # Focal length calculation: f = (W/2) / tan(FOV/2)
        self.f = (self.W / 2) / np.tan(self.FOV / 2)
        self.cam_offset = np.array([0.03, 0.0, 0.01])

    def get_relative_gate_data(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image at {image_path}")
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red spans two ranges in HSV space
        m1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        # Morphological Closing to bridge gaps in the ring
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_data = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150: continue 

            hull = cv2.convexHull(cnt)
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            
            if radius > 10:
                # Estimate depth (Z) based on known gate size vs pixel size
                dist_z = (self.GATE_RADIUS * self.f) / radius
                
                # Project pixel coordinates to camera space
                rel_x_cam = (cx - self.W/2) * dist_z / self.f
                rel_y_cam = (cy - self.H/2) * dist_z / self.f
                
                # Transform to Drone Body Frame (assuming X-forward, Y-left, Z-up)
                rel_pos = np.array([
                    dist_z + self.cam_offset[0],
                    -rel_x_cam + self.cam_offset[1],
                    -rel_y_cam + self.cam_offset[2]
                ])

                # Orientation estimation
                if len(cnt) >= 5 and area > 1000:
                    try:
                        (ecx, ecy), (MA, ma), angle_deg = cv2.fitEllipse(cnt)
                        tilt = np.arccos(np.clip(ma / MA, 0, 1))
                        phi = np.radians(angle_deg)
                        nx, ny, nz = np.cos(tilt), np.sin(tilt)*np.cos(phi), np.sin(tilt)*np.sin(phi)
                        rel_normal = np.array([nx, ny, nz])
                    except:
                        rel_normal = np.array([1.0, 0.0, 0.0])
                else:
                    rel_normal = np.array([1.0, 0.0, 0.0])

                found_data.append((rel_pos, rel_normal))
                
                # Visual Debugging
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"Dist: {dist_z:.2f}m", (int(cx), int(cy)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show the result
        cv2.imshow("Gate Detection Test", frame)
        cv2.waitKey(0) # Press any key to close the window
        cv2.destroyAllWindows()
        
        return found_data

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    detector = GateDetector()
    img_path = "/Users/adityachoudhary/Documents/Coding/autonomous-parkour-drone/controllers/main_controller/pictures/camera_349.12.png"
    results = detector.get_relative_gate_data(img_path)

    for i, (pos, norm) in enumerate(results):
        print(f"Gate {i+1}:")
        print(f"  Relative Position (x,y,z): {pos}")
        print(f"  Normal Vector: {norm}")