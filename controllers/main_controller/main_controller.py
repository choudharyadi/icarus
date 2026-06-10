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
    detector = GateDetector(drone.camera_width, drone.camera_height)

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

    print("Starting Main Autonomous Parkour Loop with Memory...")
    
    # ─── MEMORY STORAGE ───────────────────────────────────────────────────
    # Stores a list of global unique gate coordinates: [ [X1, Y1, Z1], [X2, Y2, Z2], ... ]
    remembered_gates = []
    last_search_log_time = -2.0
    # Distance threshold to consider a detection as an "already known" gate
    DUPLICATE_THRESHOLD = 3
    # ──────────────────────────────────────────────────────────────────────

    while robot.step(timestep) != -1:
        # Get drone's current global position and orientation
        current_gps = drone.gps.getValues() # [X_global, Y_global, Z_global]
        _, _, current_yaw = drone.imu.getRollPitchYaw()

        # 1. Grab camera frame and run detector
        raw = drone.camera.getImage()
        img = np.frombuffer(raw, np.uint8).reshape((drone.camera_height, drone.camera_width, 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        visible_gates = detector.get_relative_gate_data_from_frame(img_bgr)

        # 2. Process newly seen gates and convert them to global map
        if visible_gates:
            for rel_pos, _ in visible_gates:
                # Transform relative local [Fwd, Left, Up] to absolute [X, Y, Z] world positions
                cos_yaw = np.cos(current_yaw)
                sin_yaw = np.sin(current_yaw)
                
                global_x = current_gps[0] + (rel_pos[0] * cos_yaw - rel_pos[1] * sin_yaw)
                global_y = current_gps[1] + (rel_pos[0] * sin_yaw + rel_pos[1] * cos_yaw)
                global_z = current_gps[2] + rel_pos[2]
                
                detected_global_pos = np.array([global_x, global_y, global_z])

                # Check if this gate is already in our memory map
                is_duplicate = False
                for idx, known_gate_pos in enumerate(remembered_gates):
                    # Distance between newly detected gate and a remembered gate
                    if np.linalg.norm(detected_global_pos - known_gate_pos) < DUPLICATE_THRESHOLD:
                        # Close partial views are less reliable; preserve the established center.
                        if np.linalg.norm(np.array(current_gps) - known_gate_pos) >= 3.0:
                            remembered_gates[idx] = 0.75 * known_gate_pos + 0.25 * detected_global_pos
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    print(f"[MEMORY] New gate mapped at absolute: X={global_x:.2f}, Y={global_y:.2f}")
                    remembered_gates.append(detected_global_pos)

        # 3. Decision Matrix: Find the closest gate from memory map
        if remembered_gates:
            # Calculate distance from current drone position to all remembered gates
            distances = [np.linalg.norm(np.array(current_gps) - gate) for gate in remembered_gates]
            closest_idx = np.argmin(distances)
            target_global_gate = remembered_gates[closest_idx]
            target_distance = distances[closest_idx]
            clear_after_reach = False

            # --- DYNAMIC TARGETING ADJUSTMENT ---
            # If we are close to the target gate, we need to aim PAST it to cross it.
            if target_distance < 2.5:
                print(f"[PUNCH THROUGH] Approaching remembered gate ({target_distance:.2f}m away). Punching through...")
                
                # Calculate vector direction from drone to gate to know which way is "forward" through it
                direction_vector = target_global_gate - np.array(current_gps)
                direction_vector[2] = 0 # keep it flat on horizontal plane
                horizontal_distance = np.linalg.norm(direction_vector)
                if horizontal_distance < 1e-6:
                    remembered_gates.pop(closest_idx)
                    continue
                unit_direction = direction_vector / horizontal_distance
                
                # Create an exit waypoint far enough beyond the gate to fully clear its frame.
                final_target = target_global_gate + (unit_direction * 3.0)
                
                # Only remove the gate after the exit waypoint is actually reached.
                if target_distance < 0.6:
                    clear_after_reach = True
            else:
                final_target = target_global_gate
                print(f"[TRACKING MEMORY] Headed to closest mapped gate. Distance: {target_distance:.2f}m")

            if target_distance < 2.5:
                reached = drone.goto(
                    final_target,
                    threshold=0.4,
                    max_vel=0.6,
                    yaw_alignment_threshold=0.2,
                )
            else:
                reached = drone.goto(final_target, threshold=0.4)
            
            if reached:
                print("[SUCCESS] Point reached.")
                if clear_after_reach:
                    print("[MEMORY] Gate cleared! Removing from map list.")
                    remembered_gates.pop(closest_idx)

        else:
            # 4. Fallback search mode: hold position while the camera looks for a gate.
            if robot.getTime() - last_search_log_time >= 2.0:
                print("[SEARCHING] Map is empty, looking around to discover first gate...")
                last_search_log_time = robot.getTime()
            drone.stay_hover()

if __name__ == '__main__':
    main()
