import numpy as np
import time
import serial
import pynmea2
import threading

# Kalman Filter for GPS fusion
class GPSKalmanFilter:
    def __init__(self):
        self.x = np.zeros(2)  # [latitude, longitude]
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * 0.0001  # Process noise
        self.R = np.eye(2) * 0.0005  # Measurement noise
        self.I = np.eye(2)

    def update(self, gps1, gps2):
        z = (np.array(gps1) + np.array(gps2)) / 2.0
        self.P = self.P + self.Q
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K) @ self.P
        return self.x

def parse_nmea_line(line):
    try:
        msg = pynmea2.parse(line)
        if isinstance(msg, pynmea2.types.talker.GGA) and msg.gps_qual > 0:
            return msg.latitude, msg.longitude
        elif isinstance(msg, pynmea2.types.talker.RMC) and msg.status == 'A':
            return msg.latitude, msg.longitude
    except:
        return None

gps1_data = None
gps2_data = None

# GPS reading threads
def read_gps(port, update_callback):
    with serial.Serial(port, 57600, timeout=1) as ser:
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('$GP'):
                    parsed = parse_nmea_line(line)
                    if parsed:
                        update_callback(parsed)
            except Exception as e:
                print(f"Error on {port}: {e}")

def update_gps1(data):
    global gps1_data
    gps1_data = data

def update_gps2(data):
    global gps2_data
    gps2_data = data

# Start threads
threading.Thread(target=read_gps, args=('/dev/ttyACM0', update_gps1), daemon=True).start()
threading.Thread(target=read_gps, args=('/dev/ttyACM1', update_gps2), daemon=True).start()

kf = GPSKalmanFilter()

print("Fusing GPS data from two modules:")
while True:
    if gps1_data and gps2_data:
        fused = kf.update(gps1_data, gps2_data)
        print(f"Fused Position -> Latitude: {fused[0]:.8f}, Longitude: {fused[1]:.8f}")
        time.sleep(0.5)