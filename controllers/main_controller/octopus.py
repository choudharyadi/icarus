from controller import Robot
import numpy as np
from math import cos, sin, sqrt
from route_recorder import RouteRecorder

class Octopus:
    def __init__(self, robot, timestep, pid_controller):
        self.robot = robot
        self.timestep = timestep

        # Initialize motors
        self.m1_motor = robot.getDevice("m1_motor")
        self.m1_motor.setPosition(float('inf'))
        self.m1_motor.setVelocity(-1)
        self.m2_motor = robot.getDevice("m2_motor")
        self.m2_motor.setPosition(float('inf'))
        self.m2_motor.setVelocity(1)
        self.m3_motor = robot.getDevice("m3_motor")
        self.m3_motor.setPosition(float('inf'))
        self.m3_motor.setVelocity(-1)
        self.m4_motor = robot.getDevice("m4_motor")
        self.m4_motor.setPosition(float('inf'))
        self.m4_motor.setVelocity(1)
        self.motors = [self.m1_motor, self.m2_motor, self.m3_motor, self.m4_motor]

        # Initialize sensors
        self.imu = robot.getDevice("inertial_unit")
        self.imu.enable(timestep)
        self.gps = robot.getDevice("gps")
        self.gps.enable(timestep)
        self.gyro = robot.getDevice("gyro")
        self.gyro.enable(timestep)

        # Initialize lidar sensors
        self.front_lidar = robot.getDevice("range_front")
        self.front_lidar.enable(timestep)
        self.back_lidar = robot.getDevice("range_back")
        self.back_lidar.enable(timestep)
        self.left_lidar = robot.getDevice("range_left")
        self.left_lidar.enable(timestep)
        self.right_lidar = robot.getDevice("range_right")
        self.right_lidar.enable(timestep)

        # Initialize camera
        self.camera = robot.getDevice("camera")
        self.camera.enable(timestep)
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()

        # Initialize PID controller
        self.pid_controller = pid_controller

        # Constants
        self.FLYING_ATTITUDE = 3.0
        self.TARGET_THRESHOLD = 0.1
        self.POSITION_KP = 1.0
        self.MAX_VELOCITY = 1 # Reduced from 1.0 to 0.3
        self.LIDAR_THRESHOLD = 0.5  # Threshold for detecting circle passage (circle radius is 0.5m)

        # Hover state
        self.is_hovering = False
        self.current_hover_altitude = None

        # Course tracking properties
        self.total_checkpoints = 0
        self.current_checkpoint = 0
        self.current_lap = 0
        self.total_laps = 0
        self.max_laps = 3  # Varsayılan maksimum tur sayısı
        self.is_course_active = False
        self.course_start_time = 0
        self.lap_times = []

        # Route recording
        self.route_recorder = RouteRecorder()

    def hover(self, target_altitude=None):
        """
        Makes the drone hover at a specific altitude.
        If target_altitude is not provided, uses FLYING_ATTITUDE.
        """
        if target_altitude is None:
            target_altitude = self.FLYING_ATTITUDE

        print(f"Hovering at altitude: {target_altitude}m")
        past_time = self.robot.getTime()

        while self.robot.step(self.timestep) != -1:
            dt = self.robot.getTime() - past_time
            if dt <= 0:
                past_time = self.robot.getTime()
                continue

            # Get sensor readings
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]
            altitude = self.gps.getValues()[2]

            # During hover, we want zero velocity
            v_x = 0
            v_y = 0

            # Only altitude control is active during hover
            motor_power = self.pid_controller.pid(dt, 0, 0, 0, target_altitude,
                                                  roll, pitch, yaw_rate,
                                                  altitude, v_x, v_y)

            # Apply motor commands
            self.m1_motor.setVelocity(-motor_power[0])
            self.m2_motor.setVelocity(motor_power[1])
            self.m3_motor.setVelocity(-motor_power[2])
            self.m4_motor.setVelocity(motor_power[3])

            past_time = self.robot.getTime()

            # Check if we've reached the target altitude
            if abs(altitude - target_altitude) < 0.1:  # 10cm tolerance
                print(f"Reached target altitude: {target_altitude}m")
                # Wait a few steps to stabilize
                for _ in range(20):
                    self.robot.step(self.timestep)
                # Set hover state
                self.is_hovering = True
                self.current_hover_altitude = target_altitude
                return True

        return False 

    def stay_hover(self):
        """
        Performs a single step of hover control.
        Returns True if hover is maintained successfully, False if interrupted.
        """
        if not self.is_hovering or self.current_hover_altitude is None:
            print("Cannot stay hover: Drone is not in hover state!")
            return False

        # Get sensor readings
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        altitude = self.gps.getValues()[2]

        # During hover, we want zero velocity
        v_x = 0
        v_y = 0

        # Only altitude control is active during hover
        motor_power = self.pid_controller.pid(0.01, 0, 0, 0, self.current_hover_altitude,
                                              roll, pitch, yaw_rate,
                                              altitude, v_x, v_y)

        # Apply motor commands
        self.m1_motor.setVelocity(-motor_power[0])
        self.m2_motor.setVelocity(motor_power[1])
        self.m3_motor.setVelocity(-motor_power[2])
        self.m4_motor.setVelocity(motor_power[3])

        return True

    def goto(self, target_pos, kp_pos=None, threshold=None, max_vel=None):
        """
        Makes the drone go to a specific position.
        target_pos should be a list/tuple of [x, y, z]
        """
        # Reset hover state when starting to move
        self.is_hovering = False
        self.current_hover_altitude = None

        if kp_pos is None:
            kp_pos = self.POSITION_KP
        if threshold is None:
            threshold = self.TARGET_THRESHOLD
        if max_vel is None:
            max_vel = self.MAX_VELOCITY

        target_x, target_y, target_z = target_pos
        print(f"Moving to target: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")

        loop_start_time = self.robot.getTime()
        past_time = self.robot.getTime()
        past_x_global = self.gps.getValues()[0]
        past_y_global = self.gps.getValues()[1]

        # Denge kontrolü için değişkenler
        max_pitch_roll = 0.3  # Maksimum pitch ve roll açısı (radyan)
        pitch_roll_stable_count = 0  # Dengeli durum sayacı
        required_stable_steps = 5  # Gerekli dengeli adım sayısı

        while self.robot.step(self.timestep) != -1:
            # Rota noktasını kaydet
            if hasattr(self, 'route_recorder'):
                self.route_recorder.record_point(self)

            current_time = self.robot.getTime()
            dt = current_time - past_time
            if dt <= 0:
                past_time = current_time
                continue

            # Get sensor readings
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]
            x_global, y_global, altitude = self.gps.getValues()

            # Denge kontrolü
            if abs(roll) < max_pitch_roll and abs(pitch) < max_pitch_roll:
                pitch_roll_stable_count += 1
            else:
                pitch_roll_stable_count = 0
                print(f"Drone dengesiz! Roll: {roll:.2f}, Pitch: {pitch:.2f}")

            # Calculate velocities
            if dt > 1e-5:
                v_x_global = (x_global - past_x_global) / dt
                v_y_global = (y_global - past_y_global) / dt
            else:
                v_x_global = 0
                v_y_global = 0

            # Transform to body frame
            cos_yaw = cos(yaw)
            sin_yaw = sin(yaw)
            v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
            v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

            # Check if target reached
            distance_to_target = sqrt((target_x - x_global) ** 2 +
                                      (target_y - y_global) ** 2 +
                                      (target_z - altitude) ** 2)

            if distance_to_target < threshold:
                print(f"Target reached! Distance: {distance_to_target:.3f}m")
                # Set hover state at target position
                self.is_hovering = True
                self.current_hover_altitude = target_z
                return True

            # Calculate desired velocities
            error_x_global = target_x - x_global
            error_y_global = target_y - y_global

            # Calculate desired yaw angle
            desired_yaw = np.arctan2(error_y_global, error_x_global)
            
            # Normalize yaw difference to [-pi, pi]
            yaw_diff = desired_yaw - yaw
            while yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi

            # Yaw control
            yaw_desired = np.clip(yaw_diff * 0.5, -0.5, 0.5)

            # Hareket kontrolü
            if abs(yaw_diff) < 0.1:  # ~5.7 derece
                # İleri hareket
                desired_forward_speed_raw = error_x_global * cos_yaw + error_y_global * sin_yaw
                desired_sideways_speed_raw = -error_x_global * sin_yaw + error_y_global * cos_yaw
                
                # Denge düzeltmesi
                if abs(roll) > max_pitch_roll:
                    desired_sideways_speed_raw -= roll * 0.2
                if abs(pitch) > max_pitch_roll:
                    desired_forward_speed_raw -= pitch * 0.2
            else:
                # Sadece dönüş
                desired_forward_speed_raw = 0
                desired_sideways_speed_raw = 0

            forward_desired = kp_pos * desired_forward_speed_raw
            sideways_desired = kp_pos * desired_sideways_speed_raw

            # Limit velocity
            current_desired_speed = sqrt(forward_desired ** 2 + sideways_desired ** 2)
            if current_desired_speed > max_vel:
                scale = max_vel / current_desired_speed
                forward_desired *= scale
                sideways_desired *= scale

            # Get motor commands from PID controller
            motor_power = self.pid_controller.pid(dt, forward_desired, sideways_desired,
                                                  yaw_desired, target_z,
                                                  roll, pitch, yaw_rate,
                                                  altitude, v_x, v_y)

            # Apply motor commands
            self.m1_motor.setVelocity(-motor_power[0])
            self.m2_motor.setVelocity(motor_power[1])
            self.m3_motor.setVelocity(-motor_power[2])
            self.m4_motor.setVelocity(motor_power[3])

            # Update past values
            past_time = current_time
            past_x_global = x_global
            past_y_global = y_global

        return False

    def get_camera_image(self):
        """
        Returns the current camera image as bytes.
        The image is in RGB format with width and height as defined in initialization.
        """
        return self.camera.getImage()

    def capture_image(self):
        """
        Captures and saves a camera image with timestamp.
        Returns the filename of the saved image.
        """
        try:
            from PIL import Image
            import numpy as np
            import os
            
            # Create pictures directory if it doesn't exist
            if not os.path.exists('pictures'):
                os.makedirs('pictures')
            
            # Get image data (Webots returns RGBA format)
            image_data = self.get_camera_image()
            
            # Convert RGBA bytes to numpy array
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_array = image_array.reshape((self.camera_height, self.camera_width, 4))
            
            # Convert RGBA to RGB (drop alpha channel and convert BGR to RGB)
            image_bgr = image_array[:, :, :3]
            image_rgb = image_bgr[:, :, ::-1]  # Reverse color channels BGR -> RGB
            
            # Create PIL Image from numpy array
            image = Image.fromarray(image_rgb, 'RGB')
            
            # Create filename with timestamp
            timestamp = self.robot.getTime()
            filename = f'pictures/camera_{timestamp:.2f}.png'
            
            # Save image
            image.save(filename)
            print(f"\nPicture saved: {filename}")
            return filename
        except ImportError:
            print("\nError: PIL/Pillow or NumPy not installed. Install with: pip install Pillow numpy")
            return None
        except Exception as e:
            print(f"\nError capturing image: {e}")
            return None
