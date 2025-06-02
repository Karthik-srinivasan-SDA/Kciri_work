'''import sys
import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import math

sys.path.append('/home/karthik/carla_simulator/PythonAPI/carla')

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

class CameraDisplay:
    def __init__(self):
        self.display_active = True
        self.current_frame = None
    
    def callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.current_frame = array[:, :, :3]

class CenteredLHTAgent(BasicAgent):
    def __init__(self, vehicle, target_speed=20):
        super().__init__(vehicle, target_speed=target_speed)
        self._map = vehicle.get_world().get_map()
        self._lane_width_cache = {}
        self._spline_x = None
        self._spline_y = None
        self._spline_length = 0
        self._current_spline_index = 0
        self._spline_resolution = 2.0
        self._lookahead_distance = 20.0
        
    def _get_lane_width(self, waypoint):
        lane_id = (waypoint.road_id, waypoint.lane_id)
        if lane_id not in self._lane_width_cache:
            self._lane_width_cache[lane_id] = waypoint.lane_width
        return self._lane_width_cache[lane_id]
    
    def _calculate_center_position(self, waypoint):
        lane_width = self._get_lane_width(waypoint)
        yaw = np.radians(waypoint.transform.rotation.yaw)
        
        offset = lane_width * 0.5
        x_offset = -offset * np.sin(yaw)
        y_offset = offset * np.cos(yaw)
        
        return carla.Location(
            x=waypoint.transform.location.x + x_offset,
            y=waypoint.transform.location.y + y_offset,
            z=waypoint.transform.location.z
        )
    
    def _create_spline_trajectory(self, waypoints):
        """Create cubic spline trajectory from waypoints"""
        x_coords = []
        y_coords = []
        distances = [0]
        
        prev_x, prev_y = None, None
        for wp, _ in waypoints:
            if wp is None:
                continue
            current_x = wp.transform.location.x
            current_y = wp.transform.location.y
            
            if prev_x is not None and prev_y is not None:
                dx = current_x - prev_x
                dy = current_y - prev_y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < 0.1:
                    continue
                distances.append(distances[-1] + distance)
            
            x_coords.append(current_x)
            y_coords.append(current_y)
            prev_x, prev_y = current_x, current_y
        
        if len(x_coords) < 4:
            print("Not enough waypoints for cubic spline (need at least 4)")
            self._spline_x = None
            self._spline_y = None
            return
        
        if not all(x < y for x, y in zip(distances, distances[1:])):
            print("Distance sequence not strictly increasing - adjusting...")
            for i in range(1, len(distances)):
                if distances[i] <= distances[i-1]:
                    distances[i] = distances[i-1] + 0.1
        
        try:
            self._spline_x = CubicSpline(distances, x_coords)
            self._spline_y = CubicSpline(distances, y_coords)
            self._spline_length = distances[-1]
        except ValueError as e:
            print(f"Failed to create splines: {e}")
            self._spline_x = None
            self._spline_y = None
    
    def _get_spline_point(self, distance):
        if not self._spline_x or not self._spline_y:
            return None
        
        distance = max(0, min(distance, self._spline_length))
        return carla.Location(
            x=float(self._spline_x(distance)),
            y=float(self._spline_y(distance)),
            z=0
        )
    
    def set_global_plan(self, global_plan):
        corrected_plan = []
        for wp, option in global_plan:
            leftmost_wp = wp
            while True:
                next_left = leftmost_wp.get_left_lane()
                if not next_left or next_left.lane_type != carla.LaneType.Driving:
                    break
                leftmost_wp = next_left
            
            center_location = self._calculate_center_position(leftmost_wp)
            center_wp = self._map.get_waypoint(center_location)
            
            if center_wp:
                corrected_plan.append((center_wp, option))
            else:
                corrected_plan.append((wp, option))
        
        super().set_global_plan(corrected_plan)
        self._create_spline_trajectory(corrected_plan)
    
    def run_step(self):
        if not self._spline_x or not self._spline_y:
            return super().run_step()
        
        current_location = self._vehicle.get_location()
        current_velocity = self._vehicle.get_velocity()
        speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2)
        
        min_distance = float('inf')
        closest_distance_along_spline = 0
        
        search_step = max(1.0, speed * 0.1)
        for d in np.arange(0, self._spline_length, search_step):
            point = self._get_spline_point(d)
            if not point:
                continue
            dist = math.sqrt((point.x - current_location.x)**2 + 
                            (point.y - current_location.y)**2)
            if dist < min_distance:
                min_distance = dist
                closest_distance_along_spline = d
        
        lookahead_distance = min(self._lookahead_distance, 5 + speed * 0.3)
        target_distance = min(closest_distance_along_spline + lookahead_distance, 
                            self._spline_length)
        target_point = self._get_spline_point(target_distance)
        
        if not target_point:
            return super().run_step()
        
        target_waypoint = self._map.get_waypoint(target_point)
        if target_waypoint:
            target_point.z = target_waypoint.transform.location.z
        
        dx = target_point.x - current_location.x
        dy = target_point.y - current_location.y
        desired_yaw = math.degrees(math.atan2(dy, dx))
        
        current_yaw = math.degrees(
            math.atan2(current_velocity.y, current_velocity.x)) if speed > 0.1 else self._vehicle.get_transform().rotation.yaw
        
        yaw_error = (desired_yaw - current_yaw + 180) % 360 - 180
        steering = np.clip(yaw_error / 60.0, -1.0, 1.0)
        
        control = carla.VehicleControl()
        control.throttle = 0.75 if speed < self._target_speed * 0.2778 else 0.0
        control.steer = steering
        control.brake = 0.0
        
        world = self._vehicle.get_world()
        world.debug.draw_point(
            target_point, 
            size=0.2, 
            color=carla.Color(r=255, g=0, b=0), 
            life_time=0.1
        )
        world.debug.draw_line(
            current_location, 
            target_point, 
            thickness=0.05,
            color=carla.Color(r=0, g=255, b=0), 
            life_time=0.1
        )
        
        return control

def spawn_vehicle(world, blueprint_library, spawn_point):
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        print(f"Vehicle spawned at {spawn_point.location}")
        return vehicle
    print("Failed to spawn vehicle")
    return None

def setup_camera(world, blueprint_library, vehicle):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '110')
    camera_transform = carla.Transform(
        carla.Location(x=1.6, y=0.0, z=1.7),
        carla.Rotation(pitch=-15.0)
    )
    return world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

def draw_left_waypoints(world, route):
    """Visualize left lane boundaries with improved color scheme"""
    for i, (wp, _) in enumerate(route):
        # Get leftmost waypoint
        leftmost_wp = wp
        while True:
            next_left = leftmost_wp.get_left_lane()
            if not next_left or next_left.lane_type != carla.LaneType.Driving:
                break
            leftmost_wp = next_left
        
        lane_width = leftmost_wp.lane_width
        yaw = np.radians(leftmost_wp.transform.rotation.yaw)
        
        # Main boundary line (soft blue with slight transparency)
        boundary_pos = carla.Location(
            x=leftmost_wp.transform.location.x - (lane_width/2) * np.sin(yaw),
            y=leftmost_wp.transform.location.y + (lane_width/2) * np.cos(yaw),
            z=leftmost_wp.transform.location.z + 0.5
        )
        
        # Draw boundary line (semi-transparent teal)
        world.debug.draw_point(
            boundary_pos,
            color=carla.Color(r=0, g=200, b=200, a=150),
            size=0.25,
            life_time=120.0
        )
        
        # Connection lines between waypoints
        if i > 0:
            prev_wp = route[i-1][0]
            prev_leftmost = prev_wp
            while True:
                next_left = prev_leftmost.get_left_lane()
                if not next_left or next_left.lane_type != carla.LaneType.Driving:
                    break
                prev_leftmost = next_left
            
            prev_boundary = carla.Location(
                x=prev_leftmost.transform.location.x - (prev_leftmost.lane_width/2) * np.sin(np.radians(prev_leftmost.transform.rotation.yaw)),
                y=prev_leftmost.transform.location.y + (prev_leftmost.lane_width/2) * np.cos(np.radians(prev_leftmost.transform.rotation.yaw)),
                z=prev_leftmost.transform.location.z + 0.5
            )
            
            # Boundary connection (semi-transparent light blue)
            world.debug.draw_line(
                prev_boundary,
                boundary_pos,
                thickness=0.08,
                color=carla.Color(r=100, g=200, b=255, a=100),
                life_time=120.0
            )
            
        # Direction indicators every 10 waypoints
        if i % 10 == 0:
            arrow_end = carla.Location(
                x=boundary_pos.x + 1.5 * np.cos(yaw),
                y=boundary_pos.y + 1.5 * np.sin(yaw),
                z=boundary_pos.z
            )
            world.debug.draw_arrow(
                boundary_pos,
                arrow_end,
                thickness=0.05,
                arrow_size=0.2,
                color=carla.Color(r=255, g=100, b=0),
                life_time=120.0
            )

def main():
    camera_display = CameraDisplay()
    
    # Initialize CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.load_world('Town04')
    
    # Setup
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle = spawn_vehicle(world, blueprint_library, spawn_points[10])
    if not vehicle:
        return
    
    camera = setup_camera(world, blueprint_library, vehicle)
    camera.listen(camera_display.callback)
    
    # Plan route
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    route = grp.trace_route(spawn_points[10].location, spawn_points[50].location)
    
    # Initialize agent
    agent = CenteredLHTAgent(vehicle, target_speed=40)
    agent.set_global_plan(route)
    
    try:
        while camera_display.display_active:
            # Draw improved left waypoints visualization
            draw_left_waypoints(world, route)
            
            if agent.done():
                print("Reached destination")
                break
                
            control = agent.run_step()
            vehicle.apply_control(control)
            
            if camera_display.current_frame is not None:
                frame = cv2.cvtColor(camera_display.current_frame, cv2.COLOR_BGR2RGB)
                speed = vehicle.get_velocity().length() * 3.6
                
                # Updated HUD display
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "LHT - Safe Lane Navigation", (20, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
                cv2.putText(frame, "Cubic Spline Trajectory", (20, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
                
                cv2.imshow("CARLA Safe Lane Navigation", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    camera_display.display_active = False
            
            world.tick()
            
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()'''


import sys
import numpy as np
import cv2
import math
from scipy.special import comb

sys.path.append('/home/karthik/carla_simulator/PythonAPI/carla')

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

class CameraDisplay:
    def __init__(self):
        self.display_active = True
        self.current_frame = None
    
    def callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.current_frame = array[:, :, :3]

class BezierTrajectoryAgent(BasicAgent):
    def __init__(self, vehicle, target_speed=20):
        super().__init__(vehicle, target_speed=target_speed)
        self._map = vehicle.get_world().get_map()
        self._lane_width_cache = {}
        self._bezier_trajectory = []
        self._current_bezier_index = 0
        self._lookahead_distance = 20.0
        self._segment_length = 4  # Number of waypoints per Bézier segment
        self._debug_points = []
        
    def _get_lane_width(self, waypoint):
        lane_id = (waypoint.road_id, waypoint.lane_id)
        if lane_id not in self._lane_width_cache:
            self._lane_width_cache[lane_id] = waypoint.lane_width
        return self._lane_width_cache[lane_id]
    
    def _calculate_center_position(self, waypoint):
        lane_width = self._get_lane_width(waypoint)
        yaw = np.radians(waypoint.transform.rotation.yaw)
        
        offset = lane_width * 0.5
        x_offset = -offset * np.sin(yaw)
        y_offset = offset * np.cos(yaw)
        
        return carla.Location(
            x=waypoint.transform.location.x + x_offset,
            y=waypoint.transform.location.y + y_offset,
            z=waypoint.transform.location.z
        )
    
    def _generate_bezier_segment(self, waypoints):
        """Generate Bézier curve segment for a set of waypoints"""
        if len(waypoints) < 2:
            return None
            
        # Convert waypoints to numpy array
        points = np.array([(wp.transform.location.x, 
                           wp.transform.location.y,
                           wp.transform.location.z) for wp in waypoints])
        
        # For small segments, use linear interpolation
        if len(waypoints) == 2:
            t = np.linspace(0, 1, 20)
            return (1-t)[:, np.newaxis] * points[0] + t[:, np.newaxis] * points[1]
        
        # For longer segments, use cubic Bézier
        n = len(points) - 1
        t = np.linspace(0, 1, 20)
        
        # Calculate Bézier curve using Bernstein polynomials
        curve = np.zeros((len(t), 3))
        for i in range(n + 1):
            term = comb(n, i) * (t**i) * (1 - t)**(n - i)
            curve += term[:, np.newaxis] * points[i]
        
        return curve
    
    def _create_bezier_trajectory(self, waypoints):
        """Create complete Bézier trajectory from all waypoints"""
        trajectory = []
        
        # Group waypoints into segments
        for i in range(0, len(waypoints), self._segment_length-1):
            segment = waypoints[i:i+self._segment_length]
            if len(segment) < 2:
                continue
                
            # Get the waypoint objects from the tuples
            segment_waypoints = [wp for wp, _ in segment]
            
            # Generate Bézier segment
            segment_curve = self._generate_bezier_segment(segment_waypoints)
            if segment_curve is None:
                continue
                
            # Avoid duplicate points between segments
            if trajectory and len(trajectory) > 0:
                segment_curve = segment_curve[1:]
                
            trajectory.extend(segment_curve)
        
        return np.array(trajectory)
    
    def _get_closest_point_index(self, trajectory, location):
        """Find closest point on trajectory to vehicle"""
        vehicle_pos = np.array([location.x, location.y, location.z])
        distances = np.linalg.norm(trajectory - vehicle_pos, axis=1)
        return np.argmin(distances)
    
    def set_global_plan(self, global_plan):
        corrected_plan = []
        for wp, option in global_plan:
            leftmost_wp = wp
            while True:
                next_left = leftmost_wp.get_left_lane()
                if not next_left or next_left.lane_type != carla.LaneType.Driving:
                    break
                leftmost_wp = next_left
            
            center_location = self._calculate_center_position(leftmost_wp)
            center_wp = self._map.get_waypoint(center_location)
            
            if center_wp:
                corrected_plan.append((center_wp, option))
            else:
                corrected_plan.append((wp, option))
        
        super().set_global_plan(corrected_plan)
        
        # Create Bézier trajectory
        self._bezier_trajectory = self._create_bezier_trajectory(corrected_plan)
        
        # Visualize trajectory
        self._visualize_trajectory()
    
    def _visualize_trajectory(self):
        """Visualize the Bézier trajectory in CARLA"""
        world = self._vehicle.get_world()
        
        # Clear previous debug points
        for point in self._debug_points:
            point.destroy()
        self._debug_points = []
        
        # Draw trajectory
        for i in range(len(self._bezier_trajectory)-1):
            start = carla.Location(
                x=self._bezier_trajectory[i][0],
                y=self._bezier_trajectory[i][1],
                z=self._bezier_trajectory[i][2] + 0.2
            )
            end = carla.Location(
                x=self._bezier_trajectory[i+1][0],
                y=self._bezier_trajectory[i+1][1],
                z=self._bezier_trajectory[i+1][2] + 0.2
            )
            
            # Store debug objects for later cleanup
            self._debug_points.append(world.debug.draw_line(
                start, end,
                thickness=0.05,
                color=carla.Color(r=0, g=255, b=0),
                life_time=0.1
            ))
    
    def run_step(self):
        if len(self._bezier_trajectory) == 0:
            return super().run_step()
        
        # Get vehicle state
        current_location = self._vehicle.get_location()
        current_velocity = self._vehicle.get_velocity()
        speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2)
        
        # Find closest point on trajectory
        closest_idx = self._get_closest_point_index(self._bezier_trajectory, current_location)
        
        # Calculate lookahead point
        lookahead_points = int(self._lookahead_distance / max(1.0, speed * 0.1))
        target_idx = min(closest_idx + lookahead_points, len(self._bezier_trajectory)-1)
        target_point = self._bezier_trajectory[target_idx]
        
        target_loc = carla.Location(
            x=target_point[0],
            y=target_point[1],
            z=target_point[2]
        )
        
        # Get target waypoint for elevation
        target_waypoint = self._map.get_waypoint(target_loc)
        if target_waypoint:
            target_loc.z = target_waypoint.transform.location.z
        
        # Calculate control
        dx = target_loc.x - current_location.x
        dy = target_loc.y - current_location.y
        desired_yaw = math.degrees(math.atan2(dy, dx))
        
        current_yaw = math.degrees(
            math.atan2(current_velocity.y, current_velocity.x)) if speed > 0.1 else self._vehicle.get_transform().rotation.yaw
        
        yaw_error = (desired_yaw - current_yaw + 180) % 360 - 180
        steering = np.clip(yaw_error / 60.0, -1.0, 1.0)
        
        control = carla.VehicleControl()
        control.throttle = 0.75 if speed < self._target_speed * 0.2778 else 0.0
        control.steer = steering
        control.brake = 0.0
        
        # Visualization
        world = self._vehicle.get_world()
        world.debug.draw_point(
            target_loc, 
            size=0.2, 
            color=carla.Color(r=255, g=0, b=0), 
            life_time=0.1
        )
        world.debug.draw_line(
            current_location, 
            target_loc, 
            thickness=0.05,
            color=carla.Color(r=0, g=255, b=0), 
            life_time=0.1
        )
        
        return control

def spawn_vehicle(world, blueprint_library, spawn_point):
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        print(f"Vehicle spawned at {spawn_point.location}")
        return vehicle
    print("Failed to spawn vehicle")
    return None

def setup_camera(world, blueprint_library, vehicle):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '110')
    camera_transform = carla.Transform(
        carla.Location(x=1.6, y=0.0, z=1.7),
        carla.Rotation(pitch=-15.0)
    )
    return world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

def draw_left_waypoints(world, route):
    """Visualize left lane boundaries"""
    for i, (wp, _) in enumerate(route):
        leftmost_wp = wp
        while True:
            next_left = leftmost_wp.get_left_lane()
            if not next_left or next_left.lane_type != carla.LaneType.Driving:
                break
            leftmost_wp = next_left
        
        lane_width = leftmost_wp.lane_width
        yaw = np.radians(leftmost_wp.transform.rotation.yaw)
        
        boundary_pos = carla.Location(
            x=leftmost_wp.transform.location.x - (lane_width/2) * np.sin(yaw),
            y=leftmost_wp.transform.location.y + (lane_width/2) * np.cos(yaw),
            z=leftmost_wp.transform.location.z + 0.5
        )
        
        world.debug.draw_point(
            boundary_pos,
            color=carla.Color(r=0, g=200, b=200, a=150),
            size=0.25,
            life_time=120.0
        )
        
        if i > 0:
            prev_wp = route[i-1][0]
            prev_leftmost = prev_wp
            while True:
                next_left = prev_leftmost.get_left_lane()
                if not next_left or next_left.lane_type != carla.LaneType.Driving:
                    break
                prev_leftmost = next_left
            
            prev_boundary = carla.Location(
                x=prev_leftmost.transform.location.x - (prev_leftmost.lane_width/2) * np.sin(np.radians(prev_leftmost.transform.rotation.yaw)),
                y=prev_leftmost.transform.location.y + (prev_leftmost.lane_width/2) * np.cos(np.radians(prev_leftmost.transform.rotation.yaw)),
                z=prev_leftmost.transform.location.z + 0.5
            )
            
            world.debug.draw_line(
                prev_boundary,
                boundary_pos,
                thickness=0.08,
                color=carla.Color(r=100, g=200, b=255, a=100),
                life_time=120.0
            )

def main():
    camera_display = CameraDisplay()
    
    # Initialize CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.load_world('Town04')
    
    # Setup
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle = spawn_vehicle(world, blueprint_library, spawn_points[10])
    if not vehicle:
        return
    
    camera = setup_camera(world, blueprint_library, vehicle)
    camera.listen(camera_display.callback)
    
    # Plan route
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    route = grp.trace_route(spawn_points[10].location, spawn_points[50].location)
    
    # Initialize agent with Bézier trajectory
    agent = BezierTrajectoryAgent(vehicle, target_speed=40)
    agent.set_global_plan(route)
    
    try:
        while camera_display.display_active:
            # Draw left waypoints visualization
            draw_left_waypoints(world, route)
            
            if agent.done():
                print("Reached destination")
                break
                
            control = agent.run_step()
            vehicle.apply_control(control)
            
            if camera_display.current_frame is not None:
                frame = cv2.cvtColor(camera_display.current_frame, cv2.COLOR_BGR2RGB)
                speed = vehicle.get_velocity().length() * 3.6
                
                # Update HUD
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "LHT - Bézier Trajectory", (20, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
                
                cv2.imshow("CARLA LHT with Bézier Trajectory", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    camera_display.display_active = False
            
            world.tick()
            
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()