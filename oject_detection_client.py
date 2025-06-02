"""import socket
import pickle
import struct
import cv2
import carla
import numpy as np
import time

# Server address and port
server_ip = "127.0.0.1"
server_port = 5000

# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)  # Make sure CARLA is running on this port
client.set_timeout(10.0)  # Set a timeout for the connection

# Get world and spawn a camera
world = client.get_world()

# Create a blueprint for an RGB camera
blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
blueprint.set_attribute('image_size_x', '640')  # Set image width
blueprint.set_attribute('image_size_y', '480')  # Set image height

# Set a camera transform
transform = carla.Transform(carla.Location(x=1.5, y=0, z=2.5))  # Adjust position and orientation as needed

# Create a camera sensor and attach it to a vehicle or world spawn
camera = world.spawn_actor(blueprint, transform)

# Initialize a socket to connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

# Define a function to process the image data
def process_image(image):
    # Convert the raw image data into an OpenCV-compatible format
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # RGBA format
    image_rgb = array[:, :, :3]  # Discard the alpha channel (use RGB)

    # Encode the frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', image_rgb)
    
    # Serialize the image data
    data = pickle.dumps(cv2.imdecode(img_encoded, cv2.IMREAD_COLOR))

    # Send the image size and data
    message_size = struct.pack(">L", len(data))  # Pack the size of the message
    client_socket.sendall(message_size + data)

# Define the callback for the camera to capture frames
def callback(image):
    process_image(image)

# Start the camera sensor listening
camera.listen(callback)

# Main loop to keep the client running and processing frames
try:
    while True:
        time.sleep(0.1)  # Optional: sleep to prevent high CPU usage

except KeyboardInterrupt:
    print("Client interrupted, closing...")
    client_socket.close()

finally:
    # Stop the camera and cleanup
    camera.stop()
    camera.destroy()
    client_socket.close()
"""


""""
import socket
import pickle
import struct
import cv2
import carla
import numpy as np
import time
from threading import Lock

# Server address and port
server_ip = "127.0.0.1"
server_port = 5000

class CarlaClient:
    def __init__(self):
        self.client_socket = None
        self.lock = Lock()
        self.connected = False
        
    def connect_to_server(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((server_ip, server_port))
            self.connected = True
            print("Connected to YOLO server")
        except ConnectionRefusedError:
            print("Failed to connect to YOLO server")
            self.connected = False
    
    def send_image(self, image):
        if not self.connected:
            return
            
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Serialize the image
            data = pickle.dumps(image)
            
            # Pack message size and send
            with self.lock:
                message_size = struct.pack(">L", len(data))
                self.client_socket.sendall(message_size + data)
        except (ConnectionResetError, BrokenPipeError):
            print("Connection to server lost")
            self.connected = False
            self.client_socket.close()
        except Exception as e:
            print(f"Error sending image: {e}")

def main():
    # Initialize CARLA client
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(10.0)
    
    # Initialize our network client
    client = CarlaClient()
    client.connect_to_server()
    
    try:
        world = carla_client.get_world()
        
        # Set up camera
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        
        # Spawn camera on a vehicle or fixed position
        # Here we'll attach it to a spectator for simplicity
        spectator = world.get_spectator()
        transform = spectator.get_transform()
        transform.location.z += 2  # 2 meters above
        transform.rotation.pitch = -15  # Look slightly downward
        
        camera = world.spawn_actor(camera_bp, transform)
        
        # Callback for camera data
        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            client.send_image(array)
        
        # Start camera
        camera.listen(camera_callback)
        
        print("Streaming to YOLO server... Press Ctrl+C to stop")
        
        # Keep the script running
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'camera' in locals():
            camera.stop()
            camera.destroy()
        if client.connected:
            client.client_socket.close()
        print("Client shutdown complete")

if __name__ == "__main__":
    main()"""

import socket
import pickle
import struct
import cv2
import carla
import numpy as np
import time
from threading import Lock

class CarlaYOLOClient:
    def __init__(self):
        # Network setup
        self.server_ip = "127.0.0.1"
        self.server_port = 5000
        self.client_socket = None
        self.lock = Lock()
        self.connected = False
        
        # CARLA setup
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = None
        self.vehicle = None
        self.camera = None

    def connect_to_server(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((self.server_ip, self.server_port))
            self.connected = True
            print("Connected to YOLO server")
        except ConnectionRefusedError:
            print("Failed to connect to YOLO server")
            self.connected = False

    def setup_carla(self):
        self.world = self.client.get_world()
        
        # Spawn a vehicle
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter('model3')[0]  # Tesla Model 3
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        
        # Set up RGB camera attached to vehicle
        camera_bp = blueprint_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        
        # Camera transform relative to vehicle (hood view)
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),  # 1.5m forward, 2.4m up
            carla.Rotation(pitch=-15)      # Slightly downward
        )
        
        # Spawn and attach camera
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        # Add collision sensor for safety
        collision_bp = blueprint_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: print("Collision detected!"))

    def send_image(self, image):
        if not self.connected:
            return
            
        try:
            # Convert CARLA image to OpenCV format
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            image_rgb = array[:, :, :3]  # Remove alpha channel
            
            # Serialize and send
            data = pickle.dumps(image_rgb)
            with self.lock:
                message_size = struct.pack(">L", len(data))
                self.client_socket.sendall(message_size + data)
        except Exception as e:
            print(f"Error sending image: {e}")
            self.connected = False

    def run(self):
        self.connect_to_server()
        self.setup_carla()
        
        # Start camera listening
        self.camera.listen(lambda image: self.send_image(image))
        
        # Put vehicle in autopilot
        self.vehicle.set_autopilot(True)
        
        print("Vehicle camera streaming to YOLO server...")
        try:
            while True:
                # You can add vehicle control logic here if needed
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.client_socket:
            self.client_socket.close()
        print("Client shutdown complete")

if __name__ == "__main__":
    client = CarlaYOLOClient()
    client.run()