#!/usr/bin/env python3

"""
Pseudo dump (help panik): # Will load in a set of waypoints in yaml from a precorded waypoint building spree on Rtabmap - these contain: waypoint_id, (x, y, z), (quaternion i,j ,k,l) - using our subscriber script. Assume last entry is the end goal...
# waypoint_purpose/class(?) - tells the robot that to behave differently after navigating to waypoint
# Using the waypoints (assuming it does not run into the objects), it will try to navigate to the position - this can be done with inspiration from the Proportional controller from bag follower hw.
# use the same logic from the original particle_localization to store the csv using the trustability
"""

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf_transformations as tr
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid
from zed_interfaces.msg import ObjectsStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Float32
from math import sqrt, atan2, cos, sin, pi
import numpy as np
import csv
import yaml
from threading import Lock
from sklearn.cluster import DBSCAN

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # Waypoint configs
        self.declare_parameter('waypoint_file', 'waypoints.yaml')
        self.declare_parameter('waypoint_tolerance', 0.5)

        # Movement behvaiour 
        # cos we wanna see objects, we need to stop for a bit just to get good readings
        # backup safety just checks in the known map whether it is safe to move back using Occupancy grid map
        self.declare_parameter('stop_duration', 2.0)
        self.declare_parameter('backup_safety_distance', 1.0)

        # Limits
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 3.0)

        # Forwards Proportional turn is quite rigid
        # TODO: Maybe okay already - cos we tested these 
        self.declare_parameter('proportional_gain_linear', 0.5)
        self.declare_parameter('proportional_gain_angular', 1.0)

        # Backwards Proportional turn is very very sensitive in comparison
        # Move slower + less angular than forward
        # TODO: Tune - assumed to be lower than the forwards ones
        self.declare_parameter('proportional_gain_linear_backward', 0.4)
        self.declare_parameter('proportional_gain_angular_backward', 0.8)

        # Camera cutoff values for objects (approx metres)
        # Set max to high is wanna just catch all
        self.declare_parameter('min_distance', 0.3)
        self.declare_parameter('max_distance', 8.0)

        # Cluster performance
        self.declare_parameter('cluster_distance', 2.0)

        self.waypoint_file = self.get_parameter('waypoint_file').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.stop_duration = self.get_parameter('stop_duration').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.proportional_gain_linear = self.get_parameter('proportional_gain_linear').value
        self.proportional_gain_angular = self.get_parameter('proportional_gain_angular').value
        self.proportional_gain_linear_backward = self.get_parameter('proportional_gain_linear_backward').value
        self.proportional_gain_angular_backward = self.get_parameter('proportional_gain_angular_backward').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.cluster_distance = self.get_parameter('cluster_distance').value
        self.backup_safety_distance = self.get_parameter('backup_safety_distance').value

        # Waypoint and navigation state
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.state = 'NAVIGATING'  # NAVIGATING, STOPPING
        self.stop_start_time = None
        self.last_cmd_vel = Twist()

        # Object dictionary
        self.object_dict = {}
        self.object_id_counter = 0
        self.mutex = Lock()

        # Robot pose and transformations
        self.robot_pose = None
        self.robot_cov = None
        self.pose_received = False
        self.R_map_camera = None
        self.t_map_camera = None
        self.R_odom_camera = None
        self.t_odom_camera = None

        # Occupancy grid
        self.grid_map = None
        self.grid_metadata = None

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/rtabmap/localization_pose',
            self.pose_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.objects_sub = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.objects_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.odom_sub = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.odom_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            '/rtabmap/grid_map',
            self.grid_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publishers
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.shadow_pub = self.create_publisher(Marker, '/debug/object_shadow', qos_profile)
        self.distance_pub = self.create_publisher(Float32, '/debug/distance_to_object', qos_profile)
        self.path_pub = self.create_publisher(nav_msgs.msg.Path, '/planned_path', qos_profile)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)

        # Timer
        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.pose_timeout = self.create_timer(5.0, self.check_pose)

        # Load waypoints
        self.load_waypoints()

    def load_waypoints(self):
        """Load waypoints from YAML and validate against occupancy grid."""
        try:
            with open(self.waypoint_file, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
            return

        waypoints = []
        for wp in data:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(wp['position']['x'])
            pose.pose.position.y = float(wp['position']['y'])
            pose.pose.position.z = float(wp['position']['z'])
            pose.pose.orientation.x = float(wp['orientation']['x'])
            pose.pose.orientation.y = float(wp['orientation']['y'])
            pose.pose.orientation.z = float(wp['orientation']['z'])
            pose.pose.orientation.w = float(wp['orientation']['w'])
            waypoints.append({
                'id': wp['id'],
                'pose': pose,
                'action': wp['action']
            })

        # Sort by waypoint_id
        waypoints.sort(key=lambda x: x['id'])
        self.waypoints = waypoints

        # Validate waypoints
        for wp in self.waypoints:
            if self.grid_map is not None and self.grid_metadata is not None:
                if not self.is_position_free(wp['pose'].pose.position.x, wp['pose'].pose.position.y):
                    self.get_logger().warn(f"Waypoint {wp['id']} at ({wp['pose'].pose.position.x}, {wp['pose'].pose.position.y}) is occupied.")

        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")
        self.publish_path()

    def is_position_free(self, x, y):
        """Check if a position is free in the occupancy grid."""
        if self.grid_map is None or self.grid_metadata is None:
            return True  # Assume free if no grid

        # Convert world coordinates to grid indices
        i = int((x - self.grid_metadata.origin.position.x) / self.grid_metadata.resolution)
        j = int((y - self.grid_metadata.origin.position.y) / self.grid_metadata.resolution)

        if 0 <= i < self.grid_metadata.width and 0 <= j < self.grid_metadata.height:
            idx = j * self.grid_metadata.width + i
            if idx < len(self.grid_map.data):
                return self.grid_map.data[idx] < 50  # Free if value < 50
        return False

    def grid_callback(self, msg):
        """Store occupancy grid and metadata."""
        self.grid_map = msg
        self.grid_metadata = msg.info
        # Re-validate waypoints
        for wp in self.waypoints:
            if not self.is_position_free(wp['pose'].pose.position.x, wp['pose'].pose.position.y):
                self.get_logger().warn(f"Waypoint {wp['id']} at ({wp['pose'].pose.position.x}, {wp['pose'].pose.position.y}) is occupied.")

    def publish_path(self):
        """Publish waypoints as a Path message for RViz."""
        path = nav_msgs.msg.Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = [wp['pose'] for wp in self.waypoints[self.current_waypoint_idx:]]
        self.path_pub.publish(path)

    def check_pose(self):
        """Log error if no pose received."""
        if not self.pose_received:
            self.get_logger().error("No valid PoseWithCovarianceStamped received on /rtabmap/localization_pose")
        self.pose_timeout.cancel()

    def pose_callback(self, msg):
        """Update robot pose and transformations."""
        self.mutex.acquire()
        try:
            cov = np.array(msg.pose.covariance).reshape(6, 6)
            if np.any(np.diag(cov) < 0):
                self.get_logger().warn("Invalid covariance in /rtabmap/localization_pose")
                return
            self.robot_pose = msg.pose.pose
            self.robot_cov = cov
            self.pose_received = True

            q = np.array([
                self.robot_pose.orientation.x,
                self.robot_pose.orientation.y,
                self.robot_pose.orientation.z,
                self.robot_pose.orientation.w
            ])
            transform_matrix = tr.quaternion_matrix(q)
            self.R_map_camera = transform_matrix[0:3, 0:3]
            self.t_map_camera = np.array([
                self.robot_pose.position.x,
                self.robot_pose.position.y,
                self.robot_pose.position.z
            ])
        finally:
            self.mutex.release()

    def odom_callback(self, msg):
        """Update odometry transformations."""
        self.mutex.acquire()
        try:
            q = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            transform_matrix = tr.quaternion_matrix(q)
            self.R_odom_camera = transform_matrix[0:3, 0:3]
            self.t_odom_camera = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
        finally:
            self.mutex.release()

    def is_backup_safe(self, robot_pos, robot_yaw):
        """Check if backing up is safe (no objects or occupied cells behind)."""
        # Check objects
        for obj in self.object_dict.values():
            obj_pos = np.array(obj['position'][:2])
            vec = obj_pos - robot_pos
            angle_to_obj = atan2(vec[1], vec[0]) - robot_yaw
            angle_to_obj = atan2(sin(angle_to_obj), cos(angle_to_obj))
            dist = sqrt(sum(vec**2))
            if abs(angle_to_obj) > pi/2 and dist < self.backup_safety_distance:
                self.get_logger().warn(f"Object at {obj_pos} blocks backup.")
                return False

        # Check occupancy grid
        if self.grid_map is not None and self.grid_metadata is not None:
            # Check cells in a 1m arc behind the robot
            for dist in np.linspace(0.1, self.backup_safety_distance, 10):
                for angle in np.linspace(-pi/2, pi/2, 10):
                    x = robot_pos[0] - dist * cos(robot_yaw + angle)
                    y = robot_pos[1] - dist * sin(robot_yaw + angle)
                    if not self.is_position_free(x, y):
                        self.get_logger().warn(f"Occupied cell at ({x}, {y}) blocks backup.")
                        return False
        return True

    def control_callback(self):
        """Navigate to waypoints using proportional control."""
        self.mutex.acquire()
        try:
            if not self.pose_received or not self.waypoints:
                return

            if self.current_waypoint_idx >= len(self.waypoints):
                self.get_logger().info("Reached final waypoint. Stopping.")
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
                return

            current_time = self.get_clock().now().nanoseconds / 1e9
            robot_pos = np.array([self.t_map_camera[0], self.t_map_camera[1]])
            robot_q = np.array([
                self.robot_pose.orientation.x,
                self.robot_pose.orientation.y,
                self.robot_pose.orientation.z,
                self.robot_pose.orientation.w
            ])
            robot_yaw = tr.euler_from_quaternion(robot_q)[2]

            wp = self.waypoints[self.current_waypoint_idx]
            wp_pos = np.array([wp['pose'].pose.position.x, wp['pose'].pose.position.y])

            # Check if waypoint is occupied
            if not self.is_position_free(wp_pos[0], wp_pos[1]):
                self.get_logger().warn(f"Waypoint {wp['id']} is occupied. Skipping.")
                self.current_waypoint_idx += 1
                self.publish_path()
                return

            # Calculate distance and angle to waypoint
            direction = wp_pos - robot_pos
            distance = sqrt(sum(direction**2))
            angle_to_waypoint = atan2(direction[1], direction[0]) - robot_yaw
            angle_to_waypoint = atan2(sin(angle_to_waypoint), cos(angle_to_waypoint))

            # State machine
            if self.state == 'STOPPING':
                if current_time - self.stop_start_time < self.stop_duration:
                    cmd = Twist()
                    self.cmd_vel_pub.publish(cmd)
                    return
                else:
                    self.get_logger().info("Finished stopping. Advancing to next waypoint.")
                    self.state = 'NAVIGATING'
                    self.current_waypoint_idx += 1
                    self.publish_path()
                    return

            # Check if waypoint reached
            if distance < self.waypoint_tolerance:
                if wp['action'] == 'stop':
                    self.get_logger().info(f"Reached waypoint {wp['id']} (stop). Pausing for {self.stop_duration}s.")
                    self.state = 'STOPPING'
                    self.stop_start_time = current_time
                    cmd = Twist()
                    self.cmd_vel_pub.publish(cmd)
                    return
                else:
                    self.get_logger().info(f"Reached waypoint {wp['id']} (navigate). Advancing.")
                    self.current_waypoint_idx += 1
                    self.publish_path()
                    return

            # Proportional control
            cmd = Twist()
            is_backward = abs(angle_to_waypoint) > pi/2
            if is_backward and self.is_backup_safe(robot_pos, robot_yaw):
                # Backward movement
                cmd.linear.x = max(-self.proportional_gain_linear_backward * distance, -self.max_linear_velocity)
                cmd.angular.z = min(
                    max(self.proportional_gain_angular_backward * angle_to_waypoint, -self.max_angular_velocity),
                    self.max_angular_velocity
                )
            else:
                # Forward movement
                cmd.linear.x = min(self.proportional_gain_linear * distance, self.max_linear_velocity)
                cmd.angular.z = min(
                    max(self.proportional_gain_angular * angle_to_waypoint, -self.max_angular_velocity),
                    self.max_angular_velocity
                )

            # Smooth commands
            alpha = 0.5
            cmd.linear.x = alpha * cmd.linear.x + (1 - alpha) * self.last_cmd_vel.linear.x
            cmd.angular.z = alpha * cmd.angular.z + (1 - alpha) * self.last_cmd_vel.angular.z
            self.last_cmd_vel = cmd

            self.cmd_vel_pub.publish(cmd)
        finally:
            self.mutex.release()

    def objects_callback(self, msg):
        """
        Process detected objects and update dictionary.
        Uses the covariance given by the messages as zed2 object detects,
        and a position covariance to determine the confidence of a reading (via variance - trace of covariance matrix)
        """
        self.mutex.acquire()
        try:
            if self.robot_pose is None or self.R_odom_camera is None:
                self.get_logger().warn("Missing robot pose or odometry. Skipping object processing.")
                return
            new_objects = []

            for obj in msg.objects:
                pos_odom = np.array([obj.position[0], obj.position[1], obj.position[2]])
                R_odom_to_camera = self.R_odom_camera.transpose()
                t_odom_to_camera = -R_odom_to_camera @ self.t_odom_camera
                pos_camera = R_odom_to_camera @ pos_odom + t_odom_to_camera
                dist = sqrt(sum(pos_camera**2))
                if dist < self.min_distance or dist > self.max_distance:
                    continue
                pos_map = self.R_map_camera @ pos_camera + self.t_map_camera
                obj_cov = np.array(obj.position_covariance).reshape(3, 3) if len(obj.position_covariance) == 9 else np.diag(obj.position_covariance[:3])
                robot_cov_2d = np.array([
                    [self.robot_cov[0, 0], self.robot_cov[0, 1], 0],
                    [self.robot_cov[1, 0], self.robot_cov[1, 1], 0],
                    [0, 0, 0]
                ])
                combined_cov = obj_cov + self.R_map_camera @ robot_cov_2d @ self.R_map_camera.T
                variance = np.trace(combined_cov)
                new_objects.append({
                    'class': obj.label,
                    'position': pos_map.tolist(),
                    'variance': variance,
                    'raw_obj': obj
                })
                # Publish bounding box shadow for this object
                self.publish_shadow(obj_id=len(new_objects)-1, raw_obj=obj, pos_map=pos_map)

            if new_objects:
                self.get_logger().info(f"New objects: {[obj['class'] for obj in new_objects]}")
                self.cluster_and_update_objects(new_objects)

                nearest_obj = min(new_objects, key=lambda o: sqrt(sum((np.array(o['position']) - self.t_map_camera)**2)))
                distance = sqrt(sum((np.array(nearest_obj['position'][:2]) - self.t_map_camera[:2])**2))
                distance_msg = Float32(data=float(distance))
                self.distance_pub.publish(distance_msg)
        finally:
            self.mutex.release()

    def cluster_and_update_objects(self, new_objects):
        """Cluster objects by position and update dictionary."""
        positions = np.array([o['position'][:2] for o in new_objects])
        if len(positions) == 0:
            return
        clustering = DBSCAN(eps=self.cluster_distance, min_samples=1).fit(positions)
        labels = clustering.labels_
        for label in set(labels):
            if label == -1:
                continue
            cluster_objs = [o for i, o in enumerate(new_objects) if labels[i] == label]
            cluster_positions = np.array([o['position'][:2] for o in cluster_objs])
            best_match_id = None
            min_dist = float('inf')
            cluster_center = np.mean(cluster_positions, axis=0)
            for obj_id, data in self.object_dict.items():
                pos = np.array(data['position'][:2])
                dist = sqrt(sum((pos - cluster_center)**2))
                if dist < min_dist and data['class'] == cluster_objs[0]['class']:
                    min_dist = dist
                    best_match_id = obj_id
            best_obj = min(cluster_objs, key=lambda o: o['variance'])
            if best_match_id is not None and min_dist < self.cluster_distance:
                if best_obj['variance'] < self.object_dict[best_match_id]['variance']:
                    self.object_dict[best_match_id] = {
                        'class': best_obj['class'],
                        'position': best_obj['position'],
                        'variance': best_obj['variance']
                    }
            else:
                self.object_dict[self.object_id_counter] = {
                    'class': best_obj['class'],
                    'position': best_obj['position'],
                    'variance': best_obj['variance']
                }
                self.object_id_counter += 1

    def publish_shadow(self, obj_id, raw_obj, pos_map):
        """Publish a bounding box shadow marker for the object in the map frame (based on Rtabmap accurate localization)."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'object_shadows'
        marker.id = obj_id  # Unique ID based on object ID
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Line width
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue to distinguish from obj_det.py (red)

        # Extract bounding box corners
        corners_obj = raw_obj.bounding_box_3d.corners
        corners = []
        for c in corners_obj:
            # Corner position in odom frame
            corner_odom = np.array([c.kp[0], c.kp[1], c.kp[2]])
            # Transform to camera frame
            R_odom_to_camera = self.R_odom_camera.transpose()
            t_odom_to_camera = -R_odom_to_camera @ self.t_odom_camera
            corner_camera = R_odom_to_camera @ corner_odom + t_odom_to_camera
            # Transform to map frame
            corner_map = self.R_map_camera @ corner_camera + self.t_map_camera
            # Project to ground plane (z=0)
            corners.append([corner_map[0], corner_map[1], 0.0])

        # Define edges for the bounding box (12 edges)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]

        # Add edges to marker
        for edge in edges:
            point_start = Point()
            point_start.x, point_start.y, point_start.z = corners[edge[0]]
            point_end = Point()
            point_end.x, point_end.y, point_end.z = corners[edge[1]]
            marker.points.append(point_start)
            marker.points.append(point_end)

        self.shadow_pub.publish(marker)

    def save_objects_to_csv(self, filename='objects.csv'):
        """Save object dictionary to CSV."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Object_ID', 'Class', 'X', 'Y', 'Z', 'Variance'])
            for obj_id, data in self.object_dict.items():
                writer.writerow([
                    obj_id,
                    data['class'],
                    data['position'][0],
                    data['position'][1],
                    data['position'][2],
                    data['variance']
                ])

    def shutdown(self):
        """Save objects and cleanup."""
        self.save_objects_to_csv()
        self.get_logger().info("Finished writing the csv")
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
