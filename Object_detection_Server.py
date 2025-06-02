"""'''
'''

import socket
import pickle
import struct
import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes (COCO)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Server setup
server_ip = "127.0.0.1"
server_port = 5000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)
print(f"YOLOv8 server listening on {server_ip}:{server_port}")

try:
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        # Receive image data
        data = b""
        payload_size = struct.calcsize(">L")
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        image_data = data[:msg_size]
        image = pickle.loads(image_data)

        # Run YOLOv8
        results = model(image, stream=True)

        # Process results and draw on image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                label = f"{classNames[cls]} {confidence:.2f}"

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show image (in a window)
        cv2.imshow("YOLOv8 Detection", image)
        cv2.waitKey(1)  # Let OpenCV display the image

        client_socket.close()

except KeyboardInterrupt:
    print("Shutting down server.")
finally:
    # Close resources properly
    server_socket.close()
    cv2.destroyAllWindows()
"""

"""
import socket
import pickle
import struct
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOServer:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "bus", "truck", 
            "traffic light", "stop sign"  # Reduced for vehicle-relevant classes
        ]
        self.server_ip = "127.0.0.1"
        self.server_port = 5000

    def process_frame(self, frame):
        results = self.model(frame, stream=True, verbose=False)
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id < len(self.class_names):  # Only show vehicle-relevant classes
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.class_names[cls_id]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def handle_client(self, client_socket):
        data = b""
        payload_size = struct.calcsize(">L")
        
        try:
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        return
                    data += packet
                
                packed_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_size)[0]
                
                while len(data) < msg_size:
                    data += client_socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                try:
                    frame = pickle.loads(frame_data)
                    processed_frame = self.process_frame(frame)
                    cv2.imshow("Vehicle Camera - YOLOv8", processed_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                except Exception as e:
                    print(f"Frame processing error: {e}")
        
        finally:
            client_socket.close()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.server_ip, self.server_port))
            server_socket.listen(1)
            print(f"YOLO Server listening on {self.server_ip}:{self.server_port}")
            
            try:
                while True:
                    client_sock, addr = server_socket.accept()
                    print(f"Connection from {addr}")
                    self.handle_client(client_sock)
            except KeyboardInterrupt:
                print("Server shutting down...")
            finally:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    server = YOLOServer()
    server.run()"""

import socket
import pickle
import struct
import cv2
import math
import numpy as np
from ultralytics import YOLO

class YOLOServer:
    def __init__(self):
        self.model = YOLO("yolo-Weights/yolov8n.pt")  # Adjust path if needed
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]
        self.server_ip = "127.0.0.1"
        self.server_port = 5000

    def process_frame(self, frame):
        results = self.model(frame, stream=True, verbose=False)
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if 0 <= cls_id < len(self.class_names):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"{self.class_names[cls_id]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def handle_client(self, client_socket):
        data = b""
        payload_size = struct.calcsize(">L")

        try:
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        return
                    data += packet

                packed_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_size)[0]

                while len(data) < msg_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        return
                    data += packet

                frame_data = data[:msg_size]
                data = data[msg_size:]

                try:
                    frame = pickle.loads(frame_data)

                    # Ensure frame is NumPy array (if not already)
                    if not isinstance(frame, np.ndarray):
                        frame = np.array(frame)

                    processed_frame = self.process_frame(frame)
                    cv2.imshow("YOLOv8 Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Frame processing error: {e}")
        finally:
            client_socket.close()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.server_ip, self.server_port))
            server_socket.listen(5)
            print(f"YOLOv8 Server listening on {self.server_ip}:{self.server_port}")

            try:
                while True:
                    client_sock, addr = server_socket.accept()
                    print(f"Connection from {addr}")
                    self.handle_client(client_sock)
            except KeyboardInterrupt:
                print("Shutting down server...")
            finally:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    server = YOLOServer()
    server.run()

