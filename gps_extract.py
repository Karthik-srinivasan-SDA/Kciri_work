import carla
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# === Define origin GPS coordinate (must match your CARLA map) ===
origin_lat = 42.0    # example for Town01
origin_lon = -1.0

# === Set up transformers ===
# Convert WGS84 (lat/lon) to UTM/meter and vice versa
transformer_to_meters = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
transformer_to_latlon = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

# Compute the UTM (meter) coordinates of the origin
origin_x_meter, origin_y_meter = transformer_to_meters.transform(origin_lon, origin_lat)

def carla_to_gps(carla_x, carla_y):
    """
    Convert Carla (x, y) → GPS (lat, lon) using pyproj.
    """
    x_meter = origin_x_meter + carla_x
    y_meter = origin_y_meter + carla_y
    lon, lat = transformer_to_latlon.transform(x_meter, y_meter)
    return lat, lon

def gps_to_carla(lat, lon):
    """
    Convert GPS (lat, lon) → Carla (x, y) using pyproj.
    """
    x_meter, y_meter = transformer_to_meters.transform(lon, lat)
    carla_x = x_meter - origin_x_meter
    carla_y = y_meter - origin_y_meter
    return carla_x, carla_y

# === Connect to CARLA ===
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()

# === Get all waypoints ===
waypoints = map.generate_waypoints(distance=7.0)

# === Create dictionary for waypoints with GPS ===
waypoint_dict = {}

for i, wp in enumerate(waypoints):
    loc = wp.transform.location
    lat, lon = carla_to_gps(loc.x, loc.y)
    
    waypoint_dict[f'waypoint_{i}'] = {
        'carla_coords': {'x': loc.x, 'y': loc.y},
        'gps_coords': {'lat': lat, 'lon': lon},
        'transform': {
            'location': {'x': loc.x, 'y': loc.y, 'z': loc.z},
            'rotation': {
                'pitch': wp.transform.rotation.pitch,
                'yaw': wp.transform.rotation.yaw,
                'roll': wp.transform.rotation.roll
            }
        },
        'road_id': wp.road_id,
        'lane_id': wp.lane_id,
        's': wp.s
    }

# === Visualization ===
x = [wp.transform.location.x for wp in waypoints]
y = [wp.transform.location.y for wp in waypoints]

plt.figure(figsize=(12, 6))

# Plot Carla coordinates
plt.subplot(1, 2, 1)
plt.scatter(x, y, s=1)
plt.gca().invert_yaxis()
plt.title("CARLA Map Waypoints")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

# Plot GPS coordinates
lats = [waypoint_dict[f'waypoint_{i}']['gps_coords']['lat'] for i in range(len(waypoints))]
lons = [waypoint_dict[f'waypoint_{i}']['gps_coords']['lon'] for i in range(len(waypoints))]

plt.subplot(1, 2, 2)
plt.scatter(lons, lats, s=1)
plt.title("GPS Coordinates via PyProj")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.show()

# === Example Usage ===
print(f"Total waypoints stored: {len(waypoint_dict)}")
print("Example waypoint data:")
print(waypoint_dict['waypoint_0'])
