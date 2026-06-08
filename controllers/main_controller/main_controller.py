from controller import Robot
from octopus import Octopus
from pid_controller import pid_velocity_fixed_height_controller
from gateDetector import GateDetector
import numpy as np
import cv2
def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    pid_controller = pid_velocity_fixed_height_controller()
    drone = Octopus(robot, timestep, pid_controller)
    detector = GateDetector()

    print("Attempting hover...")
    if not drone.hover():
        print("Hover failed — check GPS/IMU sensors")
        return

    print("Stabilizing hover for 7 seconds...")
    start_wait_time = robot.getTime()
    while robot.step(timestep) != -1:
        drone.stay_hover()
        if robot.getTime() - start_wait_time >= 7.0:
            break

    print("Starting Main Autonomous Parkour Loop...")
    frame_count = 0

    while robot.step(timestep) != -1:
        frame_count += 1
        
        # 1. Grab camera frame and run detector
        raw = drone.camera.getImage()
        img = np.frombuffer(raw, np.uint8).reshape((drone.camera_height, drone.camera_width, 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gates = detector.get_relative_gate_data_from_frame(img_bgr)

        if gates:
            # Sort out the closest visible gate
            closest_gate = min(gates, key=lambda g: np.linalg.norm(g[0]))
            closest_gate_pos = closest_gate[0]
            closest_gate_rot = closest_gate[1]
            
            dist = np.linalg.norm(closest_gate_pos)
            print(f"[TRACKING] Gate detected! Distance: {dist:.2f}m")

            # --- DYNAMIC TARGETING ADJUSTMENT ---
            # If we are close to the gate, we must target a point BEHIND it.
            # Otherwise, the drone stops directly in the frame and gets stuck.
            if dist < 2.5:
                print("[PUNCH THROUGH] Close to gate! Projecting waypoint 2.0 meters past it...")
                # Add 2 meters to the forward (X) direction relative to the gate
                closest_gate_pos[0] += 2.0 
                
            # Fly to the calculated waypoint (Blocks until arrival)
            # We set a looser threshold (0.3m) so it keeps moving fluidly
            reached = drone.goto_local(closest_gate_pos, threshold=0.3)
            
            if reached:
                print("[SUCCESS] Cleared waypoint! Scanning for next gate...")
                
        else:
            # If no gates are visible, spin slowly in place (search mode) to find one
            print("[SEARCHING] No gates visible. Yawing to scan room...")
            # We bypass goto and command a slight yaw angle change manually
            # Or use a fallback hover to keep from drifting
            drone.stay_hover()

if __name__ == '__main__':
    main()