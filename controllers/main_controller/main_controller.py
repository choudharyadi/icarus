"""main_controller controller."""

from controller import Robot, Keyboard
from octopus import Octopus
from pid_controller import pid_velocity_fixed_height_controller
from key_controller import controller


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    keyboard = Keyboard()
    keyboard.enable(timestep)

    pid_controller = pid_velocity_fixed_height_controller()
    drone = Octopus(robot, timestep, pid_controller)

    if not drone.hover():
        print("Hover failed!")
        return

    print("Hover successful!")
    print("\n=== Drone Control System ===")
    print("Controls:")
    print("W/S: Increase/Decrease altitude")
    print("A/D: Move left/right")
    print("Arrow Up/Down: Move forward/backward")
    print("Arrow Left/Right: Turn left/right")
    print("R: Start/Stop recording checkpoints")
    print("C: Start course with checkpoints")
    print("X: Reset checkpoints")
    print("V: Visualize checkpoint data")
    print("P: Take picture")
    print("Q: Quit")
    
    while robot.step(timestep) != -1:
        if not controller(drone, keyboard, timestep):
            break

if __name__ == '__main__':
    main()