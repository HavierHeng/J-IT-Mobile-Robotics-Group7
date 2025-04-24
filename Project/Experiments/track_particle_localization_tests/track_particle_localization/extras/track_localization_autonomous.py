#!/usr/bin/env python3


# Full autonomous implementation
# Might to use the mapGraph

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf_transformations as tr
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from zed_interfaces.msg import Object, ObjectsStamped
from rtabmap_msgs.msg import MapGraph
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Float32
from math import sqrt, atan2, cos, sin
import numpy as np
import csv
from threading import Lock
from sklearn.cluster import DBSCAN
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

class ObjectTrackerDualPathAvoid(Node):
    def __init__(self):
        super().__init__('object_tracker_dual_path_avoid')

        # Either you run with the pre-recorded RTabmap DB path via `/rtabmap/mapGraph` or run off hand drawn waypoints
        self.declare_parameter('path_mode', 'waypoints')  # 'map_graph' or 'waypoints'


        # Point clustering for CSV classes - DBScan
        self.declare_parameter('min_distance', 0.5)  # Min distance for object detection (m)
        self.declare_parameter('max_distance', 5.0)  # Max distance for object detection (m)
        self.declare_parameter('cluster_distance', 2.0)  # Distance threshold for clustering (m)
        self.declare_parameter('max_linear_velocity', 1.0)  # Max linear velocity (m/s)
        self.declare_parameter('max_angular_velocity', 3.0)  # Max angular velocity (rad/s)


        # Help - there pdf wording is vague cos of "using your controller" line

        # TODO: Tune these! These affects how much our car turns/moves to try to get to a point
        # Linear P controller is used to scale the speed of our car as it approaches a goal point
        # Angular P controller is used to determine how much to turn based on orientation error - value is probs somewhat close to Part A
        self.declare_parameter('nroportional_gain_linear', 0.5)  # Linear control gain
        self.declare_parameter('proportional_gain_angular', 1.0)  # Angular control gain

        # TODO: Tune these - as path following might need to be slackened
        # For controlling path following behaviours 
        self.declare_parameter('waypoint_tolerance', 0.5)  # Distance to consider waypoint reached (m)
        self.declare_parameter('map_graph_spacing', 1.0)  # Min distance between mapGraph poses (m)
        self.declare_parameter('look_ahead_distance', 1.0)  # Look-ahead distance for mapGraph (m)
        self.declare_parameter('smoothing_interval', 0.5)  # Distance between smoothed poses (m)

        # On Observation Behaviours - Controls FSM transition for robot car state
        # TODO: Does the stop distance cause it to stop forever?
        # TODO: Backup distance might need to be tuned - it affects how much to move backwards on seeing an object
        self.declare_parameter('object_stop_distance', 1.0)  # Distance to stop for objects (m)
        self.declare_parameter('observation_time', 2.0)  # Time to observe objects (s)
        self.declare_parameter('backup_distance', 0.5)  # Distance to back up (m)
        self.declare_parameter('backup_speed', 0.5)  # Speed for backing up (m/s)
        self.declare_parameter('detour_distance', 0.5)  # Lateral detour distance (m)
        self.declare_parameter('min_turning_radius', 0.5)  # Minimum turning radius (m)

        self.path_mode = self.get_parameter('path_mode').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.cluster_distance = self.get_parameter('cluster_distance').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.proportional_gain_linear = self.get_parameter('proportional_gain_linear').value
        self.proportional_gain_angular = self.get_parameter('proportional_gain_angular').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.map_graph_spacing = self.get_parameter('map_graph_spacing').value
        self.look_ahead_distance = self.get_parameter('look_ahead_distance').value
        self.smoothing_interval = self.get_parameter('smoothing_interval').value
        self.object_stop_distance = self.get_parameter('object_stop_distance').value
        self.observation_time = self.get_parameter('observation_time').value
        self.backup_distance = self.get_parameter('backup_distance').value
        self.backup_speed = self.get_parameter('backup_speed').value
        self.detour_distance = self.get_parameter('detour_distance').value
        self.min_turning_radius = self.get_parameter('min_turning_radius').value

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

        # Path following
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.last_cmd_vel = Twist()
        self.goal_pose = None
        self.last_map_graph_count = 0

        # Object observation and avoidance
        self.nearest_object_distance = float('inf')
        self.nearest_object_pos = None
        self.state = 'NAVIGATING'  # NAVIGATING, STOPPING, BACKING_UP, DETOURING
        self.state_start_time = None
        self.detour_waypoint = None

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
        self.map_graph_sub = self.create_subscription(
            MapGraph,
            '/rtabmap/mapGraph',
            self.map_graph_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publishers
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.shadow_pub = self.create_publisher(Marker, '/debug/object_shadow', qos_profile)
        self.distance_pub = self.create_publisher(Float32, '/debug/distance_to_object', qos_profile)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos_profile)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)

        # Timers
        self.pose_timeout = self.create_timer(5.0, self.check_pose)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # Load waypoints if in waypoints mode
        if self.path_mode == 'waypoints':
            self.load_waypoints()
            self.publish_path()

    def load_waypoints(self):
        """
        Load hardcoded waypoints from file - if using hand-drawn path.
        These can be taken by `ros2 topic echo /rtabmap/goal` and then in RViz publishing 2D goal points.
        TODO: replace with JSON/YAML file loading
        """
        waypoints = [
            {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}},
            {'position': {'x': 5.0, 'y': 0.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.707, 'w': 0.707}},
            {'position': {'x': 5.0, 'y': 5.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 1.0, 'w': 0.0}},
            {'position': {'x': 0.0, 'y': 5.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.707, 'w': 0.707}},
        ]
        for wp in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = wp['position']['x']
            pose.pose.position.y = wp['position']['y']
            pose.pose.position.z = wp['position']['z']
            pose.pose.orientation.x = wp['orientation']['x']
            pose.pose.orientation.y = wp['orientation']['y']
            pose.pose.orientation.z = wp['orientation']['z']
            pose.pose.orientation.w = wp['orientation']['w']
            self.waypoints.append(pose)
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")

    def smooth_poses(self, poses):
        """
        Smooth poses using cubic spline for positions and SLERP for orientations.
        This is as the subsampling can be very choppy.
        """
        if len(poses) < 2:
            return poses

        x = [p.pose.position.x for p in poses]
        y = [p.pose.position.y for p in poses]
        quats = np.array([[p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w] for p in poses])
        dist = [0.0]
        for i in range(1, len(poses)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            dist.append(dist[-1] + sqrt(dx**2 + dy**2))
        dist = np.array(dist)

        cs_x = CubicSpline(dist, x, bc_type='clamped')
        cs_y = CubicSpline(dist, y, bc_type='clamped')
        slerp = Slerp(dist, Rotation.from_quat(quats))

        total_dist = dist[-1]
        num_points = int(total_dist / self.smoothing_interval) + 1
        t = np.linspace(0, total_dist, num_points)
        smooth_x = cs_x(t)
        smooth_y = cs_y(t)
        smooth_quats = slerp(t).as_quat()

        smoothed_poses = []
        for i in range(num_points):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = smooth_x[i]
            pose.pose.position.y = smooth_y[i]
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = smooth_quats[i, 0]
            pose.pose.orientation.y = smooth_quats[i, 1]
            pose.pose.orientation.z = smooth_quats[i, 2]
            pose.pose.orientation.w = smooth_quats[i, 3]
            smoothed_poses.append(pose)

        return smoothed_poses

    def compute_detour_waypoint(self, robot_pos, robot_yaw, object_pos):
        """
        Compute a detour waypoint to avoid the object. 
        This gives us a possibly to backtrack and avoid object.
        """
        direction = np.array([cos(robot_yaw), sin(robot_yaw)])
        perp = np.array([-direction[1], direction[0]])  # Perpendicular vector
        detour_pos = robot_pos + self.detour_distance * perp  # Lateral offset
        detour_pose = PoseStamped()
        detour_pose.header.frame_id = 'map'
        detour_pose.header.stamp = self.get_clock().now().to_msg()
        detour_pose.pose.position.x = detour_pos[0]
        detour_pose.pose.position.y = detour_pos[1]
        detour_pose.pose.position.z = 0.0
        detour_pose.pose.orientation = self.robot_pose.orientation  # Maintain current orientation

        # Check turning feasibility
        dist_to_detour = sqrt(sum((detour_pos - robot_pos)**2))
        if dist_to_detour > 0:
            required_radius = dist_to_detour / (2 * sin(atan2(perp[1], perp[0])))
            if abs(required_radius) < self.min_turning_radius:
                self.get_logger().warn("Detour requires too tight a turn. Increasing detour distance.")
                detour_pos = robot_pos + 2 * self.detour_distance * perp
                detour_pose.pose.position.x = detour_pos[0]
                detour_pose.pose.position.y = detour_pos[1]

        return detour_pose

    def map_graph_callback(self, msg):
        """
        Process mapGraph poses to create a sparse path.
        This is as in reality, the robot cannot follow the exact poses we expect it to take, so the desired pose/waypoints are slackened.
        """
        if self.path_mode != 'map_graph':
            return
        self.mutex.acquire()
        try:
            if len(msg.poses) <= self.last_map_graph_count * 1.1 and self.waypoints:
                return
            self.last_map_graph_count = len(msg.poses)

            pose_data = list(zip(msg.poses, msg.pose_ids))
            pose_data.sort(key=lambda x: x[1])
            poses = [p[0] for p in pose_data]
            max_pose_id = max(msg.pose_ids)
            goal_idx = msg.pose_ids.index(max_pose_id)
            self.goal_pose = msg.poses[goal_idx]

            subsampled_poses = []
            last_pos = None
            for pose in poses:
                if pose.header.frame_id != 'map':
                    continue
                pos = np.array([pose.pose.position.x, pose.pose.position.y])
                if last_pos is None or sqrt(sum((pos - last_pos)**2)) > self.map_graph_spacing:
                    subsampled_poses.append(pose)
                    last_pos = pos

            smoothed_poses = self.smooth_poses(subsampled_poses)
            if smoothed_poses:
                last_pos = np.array([smoothed_poses[-1].pose.position.x, smoothed_poses[-1].pose.position.y])
                goal_pos = np.array([self.goal_pose.pose.position.x, self.goal_pose.pose.position.y])
                if sqrt(sum((last_pos - goal_pos)**2)) > self.smoothing_interval:
                    smoothed_poses.append(self.goal_pose)

            if smoothed_poses:
                robot_pos = np.array([self.t_map_camera[0], self.t_map_camera[1]]) if self.t_map_camera is not None else None
                if robot_pos is not None:
                    min_dist = float('inf')
                    for i, wp in enumerate(smoothed_poses):
                        wp_pos = np.array([wp.pose.position.x, wp.pose.position.y])
                        dist = sqrt(sum((wp_pos - robot_pos)**2))
                        if dist < min_dist:
                            min_dist = dist
                            self.current_waypoint_idx = i
                else:
                    self.current_waypoint_idx = 0
                self.waypoints = smoothed_poses
                self.publish_path()
                self.get_logger().info(f"Updated {len(self.waypoints)} smoothed waypoints from mapGraph, goal pose_id: {max_pose_id}")
            else:
                self.get_logger().warn("No valid poses in mapGraph")
        finally:
            self.mutex.release()

    def publish_path(self):
        """
        Publish waypoints as a Path message for RViz visualization.
        """
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = self.waypoints
        if self.detour_waypoint:
            path.poses = [self.detour_waypoint] + path.poses[self.current_waypoint_idx:]
        self.path_pub.publish(path)

    def check_pose(self):
        if not self.pose_received:
            self.get_logger().error("No valid PoseWithCovarianceStamped received on /rtabmap/localization_pose")
        self.pose_timeout.cancel()

    def pose_callback(self, msg):
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

    def control_callback(self):
        """Compute and publish /cmd_vel to follow the current waypoint or avoid objects."""
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

            # State machine
            if self.state == 'STOPPING':
                if current_time - self.state_start_time < self.observation_time:
                    cmd = Twist()
                    self.cmd_vel_pub.publish(cmd)
                    return
                else:
                    self.get_logger().info("Finished observation. Backing up.")
                    self.state = 'BACKING_UP'
                    self.state_start_time = current_time

            if self.state == 'BACKING_UP':
                # Check for objects behind
                if self.nearest_object_pos is not None:
                    obj_vec = np.array(self.nearest_object_pos[:2]) - robot_pos
                    angle_to_obj = atan2(obj_vec[1], obj_vec[0]) - robot_yaw
                    if abs(angle_to_obj) > np.pi / 2 and self.nearest_object_distance < self.object_stop_distance:
                        self.get_logger().warn("Object detected behind. Aborting backup.")
                        self.state = 'DETOURING'
                        self.state_start_time = current_time
                        self.detour_waypoint = self.compute_detour_waypoint(robot_pos, robot_yaw, self.nearest_object_pos)
                        return

                backup_time = self.backup_distance / self.backup_speed
                if current_time - self.state_start_time < backup_time:
                    cmd = Twist()
                    cmd.linear.x = -self.backup_speed
                    self.cmd_vel_pub.publish(cmd)
                    return
                else:
                    self.get_logger().info("Finished backing up. Detouring.")
                    self.state = 'DETOURING'
                    self.state_start_time = current_time
                    self.detour_waypoint = self.compute_detour_waypoint(robot_pos, robot_yaw, self.nearest_object_pos)

            if self.state == 'DETOURING':
                if self.detour_waypoint:
                    wp = self.detour_waypoint.pose
                    wp_pos = np.array([wp.position.x, wp.position.y])
                    dist = sqrt(sum((wp_pos - robot_pos)**2))
                    if dist < self.waypoint_tolerance:
                        self.get_logger().info("Reached detour waypoint. Resuming navigation.")
                        self.state = 'NAVIGATING'
                        self.detour_waypoint = None
                        self.publish_path()
                    else:
                        direction = wp_pos - robot_pos
                        angle_to_waypoint = atan2(direction[1], direction[0]) - robot_yaw
                        angle_to_waypoint = atan2(sin(angle_to_waypoint), cos(angle_to_waypoint))


                        # This was from HW4 Part 2 bag folloer - same formulua but modified for linear and angular error
                        # Now we clamp to 1.0 for linear x, and clamp to +/-3.0 radians/s
                        cmd = Twist()
                        cmd.linear.x = min(self.proportional_gain_linear * dist, self.max_linear_velocity)
                        cmd.angular.z = min(
                            max(self.proportional_gain_angular * angle_to_waypoint, -self.max_angular_velocity),
                            self.max_angular_velocity
                        )
                        alpha = 0.5
                        cmd.linear.x = alpha * cmd.linear.x + (1 - alpha) * self.last_cmd_vel.linear.x
                        cmd.angular.z = alpha * cmd.angular.z + (1 - alpha) * self.last_cmd_vel.angular.z
                        self.last_cmd_vel = cmd
                        self.cmd_vel_pub.publish(cmd)
                        return

            # Check for nearby objects
            if self.nearest_object_distance < self.object_stop_distance and self.state == 'NAVIGATING':
                self.get_logger().info("Object nearby. Stopping for observation.")
                self.state = 'STOPPING'
                self.state_start_time = current_time
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
                return

            # Normal navigation
            if self.path_mode == 'map_graph':
                min_dist = float('inf')
                target_idx = self.current_waypoint_idx
                for i in range(self.current_waypoint_idx, min(self.current_waypoint_idx + 10, len(self.waypoints))):
                    wp = self.waypoints[i].pose
                    wp_pos = np.array([wp.position.x, wp.position.y])
                    dist = sqrt(sum((wp_pos - robot_pos)**2))
                    if dist < min_dist and dist < self.look_ahead_distance:
                        min_dist = dist
                        target_idx = i
                if min_dist < self.waypoint_tolerance:
                    self.current_waypoint_idx = target_idx + 1
                    self.get_logger().info(f"Reached mapGraph pose {self.current_waypoint_idx}")
                    return
                wp = self.waypoints[target_idx].pose
            else:
                wp = self.waypoints[self.current_waypoint_idx].pose
                wp_pos = np.array([wp.position.x, wp.position.y])
                dist = sqrt(sum((wp_pos - robot_pos)**2))
                if dist < self.waypoint_tolerance:
                    self.current_waypoint_idx += 1
                    self.get_logger().info(f"Reached waypoint {self.current_waypoint_idx}")
                    return

            wp_pos = np.array([wp.position.x, wp.position.y])
            direction = wp_pos - robot_pos
            angle_to_waypoint = atan2(direction[1], direction[0]) - robot_yaw
            angle_to_waypoint = atan2(sin(angle_to_waypoint), cos(angle_to_waypoint))

            cmd = Twist()
            cmd.linear.x = min(self.proportional_gain_linear * sqrt(sum(direction**2)), self.max_linear_velocity)
            cmd.angular.z = min(
                max(self.proportional_gain_angular * angle_to_waypoint, -self.max_angular_velocity),
                self.max_angular_velocity
            )

            if self.nearest_object_distance < self.object_stop_distance * 1.5:
                cmd.linear.x *= 0.5

            alpha = 0.5
            cmd.linear.x = alpha * cmd.linear.x + (1 - alpha) * self.last_cmd_vel.linear.x
            cmd.angular.z = alpha * cmd.angular.z + (1 - alpha) * self.last_cmd_vel.angular.z
            self.last_cmd_vel = cmd

            self.cmd_vel_pub.publish(cmd)
        finally:
            self.mutex.release()

    def objects_callback(self, msg):
        """Process detected objects, transform positions, and update dictionary."""
        self.mutex.acquire()
        try:
            if self.robot_pose is None or self.R_odom_camera is None:
                self.get_logger().warn("Missing robot pose or odometry. Skipping object processing.")
                return

            new_objects = []
            self.nearest_object_distance = float('inf')
            self.nearest_object_pos = None
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
                self.publish_shadow(pos_map)
                dist_to_robot = sqrt(sum((np.array(pos_map[:2]) - self.t_map_camera[:2])**2))
                if dist_to_robot < self.nearest_object_distance:
                    self.nearest_object_distance = dist_to_robot
                    self.nearest_object_pos = pos_map.tolist()

            if new_objects:
                self.get_logger().info(f"New objects: {[obj['class'] for obj in new_objects]}")
                self.cluster_and_update_objects(new_objects)

                nearest_obj = min(new_objects, key=lambda o: sqrt(sum((np.array(o['position']) - self.t_map_camera)**2)))
                distance = sqrt(sum((np.array(nearest_obj['position'][:2]) - self.t_map_camera[:2])**2))
                distance_msg = Float32(data=float(distance))
                self.distance_pub.publish(distance_msg)
            else:
                self.nearest_object_distance = float('inf')
                self.nearest_object_pos = None
        finally:
            self.mutex.release()

    def cluster_and_update_objects(self, new_objects):
        """Cluster objects by position and update object dictionary."""
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

    def publish_shadow(self, pos_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'objects'
        marker.id = int(str(id(pos_map))[-9:])
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        marker.pose.position.x = pos_map[0]
        marker.pose.position.y = pos_map[1]
        marker.pose.position.z = 0.0
        self.shadow_pub.publish(marker)

    def save_objects_to_csv(self, filename='objects.csv'):
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
        self.save_objects_to_csv()
        self.get_logger().info("Finished writing the csv")
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerDualPathAvoid()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
