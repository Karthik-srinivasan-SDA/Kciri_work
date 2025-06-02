import carla
import numpy as np
import time

# Initialization of the simulator and client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Vehicle and sensors setup
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
vehicle = world.spawn_actor(vehicle_bp, carla.Transform(carla.Location(x=0, y=0, z=1)))

# RGB Camera setup
rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
rgb_camera = world.spawn_actor(rgb_camera_bp, carla.Transform(carla.Location(x=1.5, y=0, z=2.5)), attach_to=vehicle)

# Depth Camera setup
depth_camera_bp = blueprint_library.find('sensor.camera.depth')
depth_camera = world.spawn_actor(depth_camera_bp, carla.Transform(carla.Location(x=1.5, y=0, z=2.5)), attach_to=vehicle)

# IMU setup
imu_bp = blueprint_library.find('sensor.other.imu')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)

# GNSS setup
gnss_bp = blueprint_library.find('sensor.other.gnss')
gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)

# Set autopilot mode to obey traffic rules
vehicle.set_autopilot(True)

# Kalman Filter initialization
x = np.array([0, 0, 0])  # Initial state vector [x, y, velocity]
P = np.eye(3)  # Initial covariance matrix for 3D state

# Define time step (in seconds)
dt = 0.1

# Measurement matrix for GNSS (3D position update)
H = np.eye(3)  # Identity matrix for position update
R = np.eye(3) * 0.1  # Measurement noise covariance

# Kalman Filter update function
def update_state(x, P, z, H, R):
    y = z - np.dot(H, x)  # Innovation (difference between measured and predicted)
    S = np.dot(np.dot(H, P), H.T) + R  # Innovation covariance
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))  # Kalman gain
    x_new = x + np.dot(K, y)
    P_new = P - np.dot(np.dot(K, H), P)
    return x_new, P_new

# Kalman Filter prediction function
def predict_state(x, u, dt):
    # Simple motion model assuming constant velocity
    x_new = x[0] + x[2] * dt
    y_new = x[1] + x[2] * dt
    v_new = u[0]  # Control input: velocity
    return np.array([x_new, y_new, v_new])

# GNSS callback to update state
def gnss_callback(gnss_data):
    global x, P
    gnss_position = np.array([gnss_data.latitude, gnss_data.longitude, gnss_data.altitude])
    x, P = update_state(x, P, gnss_position, H, R)

# IMU callback to update state
def imu_callback(imu_data):
    global x, P
    imu_accel = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z])
    imu_gyro = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z])

    # Integrating acceleration to estimate velocity
    velocity = np.linalg.norm(imu_accel) * dt  # Simplified velocity estimate
    heading = x[1] + imu_gyro[2] * dt  # Update heading based on gyroscope data

    # Control input for motion model: using estimated velocity
    u = np.array([velocity])
    x = predict_state(x, u, dt)

# Listen to sensor data
gnss_sensor.listen(gnss_callback)
imu_sensor.listen(imu_callback)

# Main simulation loop
try:
    while True:
        world.tick()  # Tick the world to simulate time

        # Handle RGB and depth camera data here (optional)
        # rgb_image = rgb_camera.listen(lambda image: process_rgb_image(image))
        # depth_image = depth_camera.listen(lambda image: process_depth_image(image))

        # Simulate car movement in autopilot mode
        time.sleep(dt)

except KeyboardInterrupt:
    pass

finally:
    # Clean up actors after use
    vehicle.destroy()
    rgb_camera.destroy()
    depth_camera.destroy()
    imu_sensor.destroy()
    gnss_sensor.destroy()
