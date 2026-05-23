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

    print("Hovering! Starting detection loop...")

    frame_count = 0
    while robot.step(timestep) != -1:
        frame_count += 1

        drone.stay_hover()

        if frame_count % 10 == 0:
            raw = drone.camera.getImage()
            img = np.frombuffer(raw, np.uint8).reshape((drone.camera_height, drone.camera_width, 4))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            detector.get_relative_gate_data_from_frame(img_bgr)

if __name__ == '__main__':
    main()