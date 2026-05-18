from controller import Keyboard
from math import sqrt
from passage_point import PassagePointFilter
from checkpoint_manager import CheckpointManager
from pathfinding import CheckpointNode, create_checkpoint_connections, find_path_between_checkpoints, is_in_potential_field
from checkpoints_charts import visualize_checkpoints
import os
import numpy as np

def controller(drone, keyboard, timestep):
    ALTITUDE_CHANGE_STEP = 0.03
    last_altitude_change_time = 0
    ALTITUDE_CHANGE_INTERVAL = 0.1

    key = keyboard.getKey()
    current_time = drone.robot.getTime()
    
    forward_desired = 0
    sideways_desired = 0
    yaw_desired = 0
    target_altitude = drone.FLYING_ATTITUDE

    if not hasattr(drone, 'last_position'):
        x, y, z = drone.gps.getValues()
        drone.last_position = (x, y, z)
        drone.last_time = drone.robot.getTime()
    
    if not hasattr(drone, 'passage_filter'):
        drone.passage_filter = PassagePointFilter()
        
    if not hasattr(drone, 'checkpoint_manager'):
        drone.checkpoint_manager = CheckpointManager()
    
    if not hasattr(drone, 'key_states'):
        drone.key_states = {}
    
    if not hasattr(drone, 'recording'):
        drone.recording = False

    if key == ord('W') and (current_time - last_altitude_change_time) >= ALTITUDE_CHANGE_INTERVAL:
        drone.FLYING_ATTITUDE += ALTITUDE_CHANGE_STEP
        last_altitude_change_time = current_time
    elif key == ord('S') and (current_time - last_altitude_change_time) >= ALTITUDE_CHANGE_INTERVAL:
        drone.FLYING_ATTITUDE -= ALTITUDE_CHANGE_STEP
        last_altitude_change_time = current_time
    elif key == ord('A'):
        sideways_desired = drone.MAX_VELOCITY
    elif key == ord('D'):
        sideways_desired = -drone.MAX_VELOCITY
    elif key == Keyboard.LEFT:
        yaw_desired = 0.5
    elif key == Keyboard.RIGHT:
        yaw_desired = -0.5
    elif key == Keyboard.UP:
        forward_desired = drone.MAX_VELOCITY
    elif key == Keyboard.DOWN:
        forward_desired = -drone.MAX_VELOCITY
    elif key == ord('R'):
        if key not in drone.key_states or not drone.key_states[key]:
            drone.recording = not drone.recording
            if drone.recording:
                print("\nRecording checkpoints...")
            else:
                print("\nStopped recording checkpoints.")
        drone.key_states[key] = True
    elif key == ord('C'):
        if key not in drone.key_states or not drone.key_states[key]:
            if os.path.exists("checkpoints.json"):
                print("\nStarting course with recorded checkpoints...")
                start_course_with_checkpoints(drone)
            else:
                print("\nNo checkpoint data found. Please record checkpoints first.")
        drone.key_states[key] = True
    elif key == ord('X'):
        if key not in drone.key_states or not drone.key_states[key]:
            reset_checkpoints(drone)
        drone.key_states[key] = True
    elif key == ord('V'):
        if key not in drone.key_states or not drone.key_states[key]:
            visualize_checkpoints()
        drone.key_states[key] = True
    elif key == ord('P'):
        if key not in drone.key_states or not drone.key_states[key]:
            drone.capture_image()
        drone.key_states[key] = True
    elif key == ord('Q'):
        return False
    else:
        for k in drone.key_states:
            if k != key:
                drone.key_states[k] = False


    roll, pitch, yaw = drone.imu.getRollPitchYaw()
    yaw_rate = drone.gyro.getValues()[2]
    altitude = drone.gps.getValues()[2]

    motor_power = drone.pid_controller.pid(timestep/1000.0, forward_desired, sideways_desired,
                                        yaw_desired, target_altitude,
                                        roll, pitch, yaw_rate,
                                        altitude, 0, 0)

    drone.m1_motor.setVelocity(-motor_power[0])
    drone.m2_motor.setVelocity(motor_power[1])
    drone.m3_motor.setVelocity(-motor_power[2])
    drone.m4_motor.setVelocity(motor_power[3])

    if drone.recording:
        passage_data = detect_circle_passage(drone, drone.passage_filter)
        if passage_data:
            drone.last_position = (passage_data['position']['x'], 
                                 passage_data['position']['y'], 
                                 passage_data['position']['z'])
            drone.last_time = drone.robot.getTime()

    return True

def detect_circle_passage(drone, passage_filter):
    left_distance = drone.left_lidar.getValue() / 1000
    right_distance = drone.right_lidar.getValue() / 1000
    
    if left_distance < drone.LIDAR_THRESHOLD or right_distance < drone.LIDAR_THRESHOLD:
     
        x, y, z = drone.gps.getValues()
        roll, pitch, yaw = drone.imu.getRollPitchYaw()
        
        v_x = (x - drone.last_position[0]) / (drone.robot.getTime() - drone.last_time) if hasattr(drone, 'last_position') else 0
        v_y = (y - drone.last_position[1]) / (drone.robot.getTime() - drone.last_time) if hasattr(drone, 'last_position') else 0
        v_z = (z - drone.last_position[2]) / (drone.robot.getTime() - drone.last_time) if hasattr(drone, 'last_position') else 0
        
        side = "left" if left_distance < right_distance else "right"
        
        passage_data = {
            'position': {
                'x': x,
                'y': y,
                'z': z
            },
            'orientation': {
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw
            },
            'velocity': {
                'v_x': v_x,
                'v_y': v_y,
                'v_z': v_z
            },
            'passage_info': {
                'side': side,
                'left_lidar': left_distance,
                'right_lidar': right_distance,
                'timestamp': drone.robot.getTime()
            }
        }
        
        best_point = passage_filter.add_point(passage_data)
        if best_point:
            
            if hasattr(drone, 'last_checkpoint_position'):
                last_x, last_y, last_z = drone.last_checkpoint_position
                distance = sqrt((x - last_x)**2 + (y - last_y)**2 + (z - last_z)**2)
                if distance < 1.0:  
                    return None
            
        
            checkpoint_id = drone.checkpoint_manager.add_checkpoint(
                best_point['position'],
                best_point,
                orientation=(roll, pitch, yaw)  
            )
            if checkpoint_id is not None:
                print(f"\nNew checkpoint added with ID: {checkpoint_id}")
                print(f"Position: X={best_point['position']['x']:.2f}, Y={best_point['position']['y']:.2f}, Z={best_point['position']['z']:.2f}")
                print(f"Orientation: Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}")
                print(f"Velocity: Vx={best_point['velocity']['v_x']:.2f}, Vy={best_point['velocity']['v_y']:.2f}, Vz={best_point['velocity']['v_z']:.2f}")
                print(f"Passing from {best_point['passage_info']['side']} side")
                

                drone.last_checkpoint_position = (x, y, z)
                return best_point
        
        return None
    
    return None

def reset_checkpoints(drone):
    if os.path.exists("checkpoints.json"):
        try:
            os.remove("checkpoints.json")
            print("Checkpoint file has been deleted.")
        except Exception as e:
            print(f"Error deleting checkpoint file: {e}")
            return False

    if hasattr(drone, 'checkpoint_manager'):
        drone.checkpoint_manager = CheckpointManager()
        print("Checkpoint manager has been reset.")
    
    if hasattr(drone, 'passage_filter'):
        drone.passage_filter = PassagePointFilter()
        print("Passage filter has been reset.")
    
    if hasattr(drone, 'last_checkpoint_position'):
        delattr(drone, 'last_checkpoint_position')
        
    return True

def start_course_with_checkpoints(drone):
    checkpoints = []
    for cp_id, cp in drone.checkpoint_manager.get_all_checkpoints().items():
        checkpoint = CheckpointNode(
            int(cp_id),
            (cp['position']['x'], cp['position']['y'], cp['position']['z']),
            (
                cp['passage_history'][-1]['orientation']['roll'] if cp['passage_history'] else 0.0,
                cp['passage_history'][-1]['orientation']['pitch'] if cp['passage_history'] else 0.0,
                cp['passage_history'][-1]['orientation']['yaw'] if cp['passage_history'] else 0.0
            )
        )
        if cp['passage_history']:
            last_passage = cp['passage_history'][-1]
            checkpoint.potential_field['lidar_left'] = last_passage['lidar_readings']['left']
            checkpoint.potential_field['lidar_right'] = last_passage['lidar_readings']['right']
        checkpoints.append(checkpoint)

    if not checkpoints:
        print("Checkpoint bulunamadı!")
        return

    create_checkpoint_connections(checkpoints)

    first_checkpoint = min(checkpoints, key=lambda x: x.id)
    last_checkpoint = max(checkpoints, key=lambda x: x.id)
    
    avg_height = sum(cp.position[2] for cp in checkpoints) / len(checkpoints)
    
    if not drone.hover(avg_height):
        print("Yükseklik ayarlanamadı!")
        return

    drone.total_checkpoints = len(checkpoints)
    drone.current_checkpoint = 0
    drone.current_lap = 1
    drone.total_laps = 0
    drone.is_course_active = True
    drone.course_start_time = drone.robot.getTime()
    drone.lap_times = []

    print("\nSürekli tur sistemi başlatıldı!")
    print(f"Maksimum tur sayısı: {drone.max_laps}")
    print("Durdurmak için 'Q' tuşuna basın.")
    
    while drone.current_lap <= drone.max_laps:
        drone.route_recorder.start_recording(drone.current_lap)
        lap_start_time = drone.robot.getTime()
        
        path = find_path_between_checkpoints(first_checkpoint.id, last_checkpoint.id, checkpoints)
        if path is None:
            print("Hata: Checkpoint'ler arasında yol bulunamadı!")
            return
        print("Normal rota kullanılıyor...")
        
        for i, checkpoint in enumerate(path):
            current_pos = drone.gps.getValues()
            
            if is_in_potential_field(current_pos, checkpoint):
                continue
            
            target_pos = (checkpoint.position[0], checkpoint.position[1], avg_height)
            

            if not drone.goto(target_pos):
                print(f"Checkpoint {checkpoint.id}'e ulaşılamadı!")
                return
            
            drone.current_checkpoint = checkpoint.id
            print(f"\nTur: {drone.current_lap}/{drone.max_laps}")
            print(f"Checkpoint: {drone.current_checkpoint}/{drone.total_checkpoints}")
            
            checkpoint_start_time = drone.robot.getTime()
            stabilization_time = float(checkpoint.stabilization_time)
            photo_taken = False
            
            while drone.robot.getTime() - checkpoint_start_time < stabilization_time:
                drone.robot.step(drone.timestep)
                drone.route_recorder.record_point(drone)
                
                # Take photo at midpoint of stabilization
                if not photo_taken and (drone.robot.getTime() - checkpoint_start_time) > stabilization_time / 2:
                    drone.capture_image()
                    photo_taken = True
        
        current_time = drone.robot.getTime()
        lap_time = current_time - lap_start_time
        drone.lap_times.append(lap_time)
        

        drone.route_recorder.stop_recording()
        stats = drone.route_recorder.get_lap_statistics()
        
        print(f"\nTur {drone.current_lap} tamamlandı!")
        print(f"Tur süresi: {lap_time:.2f} saniye")
        print(f"Ortalama hız: {stats['average_speed']:.2f} m/s")
        print(f"Checkpoint süreleri: {stats['checkpoint_times']}")
        print(f"Toplam kayıt noktası: {stats['number_of_points']}")
        

        if drone.current_lap < drone.max_laps:
            print("Yeni tur başlatılıyor...")
            drone.current_lap += 1
            drone.total_laps += 1
            wait_start_time = drone.robot.getTime()
            wait_time = 1.0 
            
            while drone.robot.getTime() - wait_start_time < wait_time:
                drone.robot.step(drone.timestep)
                drone.route_recorder.record_point(drone)
        else:
            print("\nTüm turlar tamamlandı!")
            drone.is_course_active = False
            return
