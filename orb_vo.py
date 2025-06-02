'''import numpy as np
import carla
import cv2
import time
from collections import deque

# === Sensor Fusion EKF ===
class SensorFusionEKF:
    def __init__(self):
        self.state = np.zeros(9)  # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.P = np.eye(9) * 0.1
        self.Q = np.eye(9) * 0.01
        self.R = np.eye(6) * 0.5  # For visual + IMU (position + acceleration)
        self.H = np.zeros((6, 9))
        self.H[:, :6] = np.eye(6)  # Observe pos (from vision) and acc (from IMU)
        self.I = np.eye(9)

    def predict(self, dt):
        F = np.eye(9)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def get_state(self):
        return self.state

# === CARLA Setup ===
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Enable autopilot
vehicle.set_autopilot(True)

# Add IMU sensor
imu_bp = blueprint_library.find('sensor.other.imu')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=1.0)), attach_to=vehicle)

# Add camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '90')
camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

# === Sensor Data Queues ===
imu_data = None
image_queue = deque(maxlen=2)

def imu_callback(data):
    global imu_data
    imu_data = data

def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    image_queue.append(gray)

imu_sensor.listen(imu_callback)
camera.listen(camera_callback)

# === EKF Loop ===
ekf = SensorFusionEKF()
prev_time = time.time()
orb = cv2.ORB_create()

try:
    while True:
        if imu_data is None or len(image_queue) < 2:
            continue

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time

        # Get IMU acceleration
        acc = np.array([
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z
        ]) * 9.81  # Convert g to m/s²

        # Visual Odometry with ORB
        prev_gray = image_queue[0]
        curr_gray = image_queue[1]

        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        visual_pos = np.zeros(3)  # Default if not enough features

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 10:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])  # Approximate intrinsic matrix
                E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                    visual_pos = t.ravel()

        # Sensor Fusion
        z = np.concatenate((visual_pos, acc))
        ekf.predict(dt)
        ekf.update(z)
        state = ekf.get_state()

        print(f"Position: {state[:3]}, Velocity: {state[3:6]}, Orientation: {state[6:9]}")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    imu_sensor.stop()
    camera.stop()
    imu_sensor.destroy()
    camera.destroy()
    vehicle.destroy()'''

import numpy as np
import carla
import cv2
import time
from collections import deque

# === Sensor Fusion EKF ===
class SensorFusionEKF:
    def __init__(self):
        self.state = np.zeros(9)  # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.P = np.eye(9) * 0.1
        self.Q = np.eye(9) * 0.01
        self.R = np.eye(6) * 0.5  # For visual + IMU (position + acceleration)
        self.H = np.zeros((6, 9))
        self.H[:, :6] = np.eye(6)  # Observe pos (from vision) and acc (from IMU)
        self.I = np.eye(9)

    def predict(self, dt):
        F = np.eye(9)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def get_state(self):
        return self.state

# === CARLA Setup ===
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Enable autopilot using custom Traffic Manager port
tm_port = 8001
tm = client.get_trafficmanager(tm_port)
vehicle.set_autopilot(True, tm_port)

# Add IMU sensor
imu_bp = blueprint_library.find('sensor.other.imu')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=1.0)), attach_to=vehicle)

# Add camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '90')
camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

# === Sensor Data Queues ===
imu_data = None
image_queue = deque(maxlen=2)

def imu_callback(data):
    global imu_data
    imu_data = data

def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    image_queue.append(gray)

imu_sensor.listen(imu_callback)
camera.listen(camera_callback)

# === EKF Loop ===
ekf = SensorFusionEKF()
prev_time = time.time()
orb = cv2.ORB_create()

try:
    while True:
        if imu_data is None or len(image_queue) < 2:
            time.sleep(0.01)
            continue

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time

        # Get IMU acceleration
        acc = np.array([
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z
        ]) * 9.81  # Convert g to m/s²

        # Visual Odometry with ORB
        prev_gray = image_queue[0]
        curr_gray = image_queue[1]

        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        visual_pos = np.zeros(3)  # Default if not enough features

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 10:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])  # Approximate intrinsic matrix
                E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                    visual_pos = t.ravel()

        # Sensor Fusion
        z = np.concatenate((visual_pos, acc))
        ekf.predict(dt)
        ekf.update(z)
        state = ekf.get_state()

        print(f"Position: {state[:3]}, Velocity: {state[3:6]}, Orientation: {state[6:9]}")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    imu_sensor.stop()
    imu_sensor.destroy()
    camera.stop()
    camera.destroy()
    vehicle.destroy()

