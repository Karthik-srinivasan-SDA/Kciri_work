import numpy as np
import math
import time
import cv2
import pyzed.sl as sl
from pymavlink import mavutil
from datetime import datetime

# Extended Kalman Filter for Sensor Fusion
class SensorFusionEKF:
    def __init__(self):
        self.state = np.zeros(9)  # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.P = np.eye(9) * 0.1  # State covariance
        self.Q = np.eye(9) * 0.01  # Process noise
        self.R = np.eye(6) * 0.5   # Measurement noise for visual odometry and IMU
        self.H = np.zeros((6, 9))

        self.H[:, :6] = np.eye(6)

        #self.H[:6, :] = np.eye(6)  # Observe position and velocity
        self.I = np.eye(9)

    def predict(self, dt):
        # State transition model
        F = np.eye(9)
        F[0, 3] = dt  # x depends on vx
        F[1, 4] = dt  # y depends on vy
        F[2, 5] = dt  # z depends on vz

        self.state = F @ self.state  # Predict state
        self.P = F @ self.P @ F.T + self.Q  # Predict covariance

    def update(self, z):
        # Kalman Gain
        y = z - self.H @ self.state  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y  # Update state
        self.P = (self.I - K @ self.H) @ self.P  # Update covariance

    def get_state(self):
        return self.state


# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Depth mode
init_params.coordinate_units = sl.UNIT.METER       # Use meters for depth
init_params.camera_resolution = sl.RESOLUTION.HD720
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("ZED Camera not found!")
    exit(1)

# Initialize IMU (Pixhawk connection)
connection = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
connection.wait_heartbeat()
print("Connected to Pixhawk!")

# Initialize EKF
ekf = SensorFusionEKF()
prev_time = time.time()

# ORB for visual odometry
orb = cv2.ORB_create()
image = sl.Mat()
prev_gray = None

# Camera intrinsic parameters (replace with ZED calibration data)
focal_length = 529  # Example focal length
principal_point = (633, 369)
K = np.array([[focal_length, 0, principal_point[0]],
              [0, focal_length, principal_point[1]],
              [0, 0, 1]])  # Camera matrix

while True:
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # --- IMU Data from Pixhawk ---
    imu_message = connection.recv_match(type='RAW_IMU', blocking=True)
    if imu_message:
        imu_acc = np.array([imu_message.xacc, imu_message.yacc, imu_message.zacc]) * 9.81 / 1000
        imu_gyro = np.array([imu_message.xgyro, imu_message.ygyro, imu_message.zgyro]) / 1000
    else:
        continue

    # --- Visual Odometry from ZED ---
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        curr_frame = image.get_data()

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Detect ORB keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(prev_gray, None)
            kp2, des2 = orb.detectAndCompute(curr_gray, None)

            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched points
            points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate Essential Matrix and Pose
            E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

            # Update pose
            visual_position = t.ravel()
            visual_orientation = R.ravel()  # Rotation matrix
        else:
            visual_position = np.zeros(3)
            visual_orientation = np.zeros(3)

        prev_gray = curr_gray
    else:
        visual_position = np.zeros(3)
        visual_orientation = np.zeros(3)

    # --- Sensor Fusion with EKF ---
    z = np.concatenate((visual_position, imu_acc))  # Combine visual position and IMU acceleration
    ekf.predict(dt)
    ekf.update(z)
    state = ekf.get_state()

    # Extract position, velocity, and orientation from the EKF state
    pos = state[:3]
    vel = state[3:6]
    ori = state[6:9]

    print(f"Position: {pos} [m]")
    print(f"Velocity: {vel} [m/s]")
    print(f"Orientation: Roll = {ori[0]:.2f}, Pitch = {ori[1]:.2f}, Yaw = {ori[2]:.2f} [rad]")

    time.sleep(0.1)