#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import tf_transformations as tr
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Twist
from sensor_msgs.msg import PointCloud2, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from zed_interfaces.msg import Object, ObjectsStamped
import sensor_msgs.point_cloud2 as point_cloud2 from math import sqrt, cos, sin, pi, atan2
from threading import Thread, Lock
import numpy as np
from scipy.spatial import KDTree
import pickle
import cv2
import csv
import os

class Particle:
    def __init__(self, id, x, y, theta):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta

    def copy(self):
        return Particle(self.id, self.x, self.y, self.theta)

class ParticleFilter:
    def __init__(self, num_particles, map_points, ogm, xmin, xmax, ymin, ymax,
                 dynamics_translation_noise_std_dev,
                 dynamics_orientation_noise_std_dev,
                 point_cloud_measurement_noise_std_dev, node, n_thresh=0):
        self.num_particles = num_particles
        self.n_thresh = n_thresh
        self.map_points = map_points
        self.ogm = ogm
        self.grid_map = None
        self.grid_bin = None
        if ogm:
            self.grid_map = np.array(ogm.data, dtype='int8').reshape(ogm.info.height, ogm.info.width)
            self.grid_bin = (self.grid_map == 0).astype('uint8')  # Free cells
            # Filter map points based on OGM
            filtered_points = []
            for p in map_points:
                row, col = self.metric_to_grid_coords(p[0], p[1], ogm)
                if row < ogm.info.height and col < ogm.info.width and self.grid_bin[row, col]:
                    filtered_points.append(p)
            self.map_points = np.array(filtered_points) if filtered_points else map_points
        self.map_kdtree = KDTree(self.map_points)

        # From bounds of RTabmap db
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Configure dynamics uncertainties
        self.dynamics_translation_noise_std_dev = dynamics_translation_noise_std_dev
        self.dynamics_orientation_noise_std_dev = dynamics_orientation_noise_std_dev
        self.point_cloud_measurement_noise_std_dev = point_cloud_measurement_noise_std_dev

        self.eval_points = 100
        self.last_robot_odom = None
        self.robot_odom = None
        self.dx = 0
        self.dy = 0
        self.dyaw = 0
        self.particles = []
        self.weights = []
        self.node = node
        # Camera parameters
        self.fx = 700.0
        self.fy = 700.0
        self.cx = 672.0
        self.cy = 376.0
        self.img_width = 1280
        self.img_height = 720
        self.min_depth = node.get_parameter('min_depth').value
        self.max_depth = node.get_parameter('max_depth').value
        self.roi_height_ratio_top = 0.57
        self.roi_height_ratio_rect = 0.7
        self.roi_top_width_ratio = 0.2
        # Object tracking
        self.object_dict = {}
        self.min_distance = node.get_parameter('min_distance').value
        self.max_distance = node.get_parameter('max_distance').value
        self.camera_to_world = None

    def get_random_free_state(self):
        """Initialize particles in free OGM cells, facing forward."""
        while True:
            xrand = np.random.uniform(self.xmin, self.xmax)
            yrand = np.random.uniform(self.ymin, self.ymax)
            if self.grid_bin is not None:
                row, col = self.metric_to_grid_coords(xrand, yrand, self.ogm)
                if row >= self.ogm.info.height or col >= self.ogm.info.width or not self.grid_bin[row, col]:
                    continue
            dist, _ = self.map_kdtree.query([xrand, yrand, 0])
            if dist > 0.1:
                theta = 0.0  # Facing straight
                return xrand, yrand, theta

    def init_particles(self):
        """Initialize particles with prior knowledge."""
        for i in range(self.num_particles):
            xrand, yrand, theta = self.get_random_free_state()
            self.particles.append(Particle(i, xrand, yrand, theta))
        self.weights = np.ones(self.num_particles) / self.num_particles

    @staticmethod
    def exponentiate_and_normalize(x):
        b = x.max()
        x = np.exp(x - b)
        return x / x.sum()

    def create_roi_mask(self, height, width):
        """Create trapezoidal ROI mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        rect_top_y = int(height * self.roi_height_ratio_rect)
        cv2.rectangle(mask, (0, rect_top_y), (width, height), 255, -1)
        top_width = int(width * self.roi_top_width_ratio)
        bottom_width = width
        top_y = int(height * self.roi_height_ratio_top)
        bottom_y = rect_top_y
        trapezoid = np.array([
            [(width - top_width) // 2, top_y],
            [(width + top_width) // 2, top_y],
            [(width + bottom_width) // 2, bottom_y],
            [(width - bottom_width) // 2, bottom_y]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [trapezoid], 255)
        return mask

    def filter_points_by_frustum(self, x, y, theta):
        """Filter map points within the camera frustum for a particle's pose."""
        # Pre-filter points using KDTree
        query_point = np.array([x, y, 0])
        radius = self.max_depth + 1.0  # Margin for robustness
        indices = self.map_kdtree.query_ball_point(query_point, radius, p=2)
        nearby_points = self.map_points[indices]
        if not nearby_points.any():
            return np.array([])
        # Transform points to camera frame
        R = np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
        t = np.array([x, y, 0])
        filtered_points = []
        for p in nearby_points:
            p_camera = R.transpose() @ (p - t)
            x_c, y_c, z_c = p_camera
            if z_c < self.min_depth or z_c > self.max_depth:
                continue
            # Project to image plane
            u = self.fx * x_c / z_c + self.cx
            v = self.fy * y_c / z_c + self.cy
            if 0 <= u < self.img_width and 0 <= v < self.img_height:
                filtered_points.append(p_camera)
        return np.array(filtered_points) if filtered_points else np.array([])

    def subsample_point_cloud(self, point_cloud_msg):
        """Subsample point cloud and apply ROI."""
        points = []
        for p in point_cloud2.read_points(point_cloud_msg, field_names=("x", "y", "z")):
            x, y, z = p
            if z < self.min_depth or z > self.max_depth:
                continue
            u = int(self.fx * x / z + self.cx)
            v = int(self.fy * y / z + self.cy)
            if 0 <= u < self.img_width and 0 <= v < self.img_height:
                points.append([x, y, z])
        if not points:
            return [], []
        points = np.array(points)
        mask = self.create_roi_mask(self.img_height, self.img_width)
        valid_points = []
        pixel_coords = []
        for p in points:
            x, y, z = p
            u = int(self.fx * x / z + self.cx)
            v = int(self.fy * y / z + self.cy)
            if 0 <= u < self.img_width and 0 <= v < self.img_height and mask[v, u] > 0:
                valid_points.append(p)
                pixel_coords.append([u, v])
        if len(valid_points) > self.eval_points:
            indices = np.random.choice(len(valid_points), self.eval_points, replace=False)
            valid_points = np.array(valid_points)[indices]
            pixel_coords = np.array(pixel_coords)[indices]
        return valid_points, pixel_coords

    def project_to_ogm(self, points, ogm):
        """Project 3D points to 2D OGM for simplified matching."""
        projected_points = []
        for p in points:
            row, col = self.metric_to_grid_coords(p[0], p[1], ogm)
            if row < ogm.info.height and col < ogm.info.width and self.grid_bin[row, col]:
                projected_points.append([p[0], p[1]])
        return np.array(projected_points) if projected_points else np.array([])

    def simulate_point_cloud_for_particle(self, x, y, theta, points):
        """Transform observed points to world frame, simulating particle's view."""
        # Get frustum-filtered map points
        visible_map_points = self.filter_points_by_frustum(x, y, theta)
        if not visible_map_points.any():
            return np.array([])
        # Transform observed points to world frame
        transformed_points = []
        R = np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
        t = np.array([x, y, 0])
        for p in points:
            p_world = R @ p + t
            transformed_points.append(p_world)
        return np.array(transformed_points) if transformed_points else np.array([])

    def get_prediction_error_squared(self, point_cloud_msg, particle):
        """Compute error between observed and map point clouds."""
        if particle.x < self.xmin or particle.x > self.xmax or particle.y < self.ymin or particle.y > self.ymax:
            return 1e6
        if self.grid_bin is not None:
            row, col = self.metric_to_grid_coords(particle.x, particle.y, self.ogm)
            if row < self.ogm.info.height and col < self.ogm.info.width and not self.grid_bin[row, col]:
                return 1e6
        actual_points, _ = self.subsample_point_cloud(point_cloud_msg)
        if not actual_points.any():
            return 1e6
        # Optional: Project to OGM for simplified matching
        if self.ogm:
            actual_points_2d = self.project_to_ogm(actual_points, self.ogm)
            if not actual_points_2d.any():
                return 1e6
            actual_points = np.hstack((actual_points_2d, np.zeros((actual_points_2d.shape[0], 1))))
        predicted_points = self.simulate_point_cloud_for_particle(particle.x, particle.y, particle.theta, actual_points)
        if not predicted_points.any():
            return 1e6
        # Use frustum-filtered map points
        visible_map_points = self.filter_points_by_frustum(particle.x, particle.y, particle.theta)
        if not visible_map_points.any():
            return 1e6
        errors = []
        for p in predicted_points:
            dist, _ = self.map_kdtree.query(p)
            errors.append(dist ** 2)
        return sum(errors) / len(errors) if errors else 1e6

    def handle_observation(self, point_cloud_msg, dt):
        """Update particle weights and resample."""
        errors = []
        for particle in self.particles:
            self.predict_particle_odometry(particle, dt)
            error = self.get_prediction_error_squared(point_cloud_msg, particle)
            errors.append(error)
        errors = np.array(errors)
        neg_errors = -errors / (2 * self.point_cloud_measurement_noise_std_dev ** 2)
        self.weights = self.exponentiate_and_normalize(neg_errors)
        N_eff = 1.0 / np.sum(np.square(self.weights))
        self.node.get_logger().info(f"Effective Sample Size: {N_eff}")
        if N_eff < self.n_thresh or self.n_thresh == 0:
            self.resample()

    def resample(self):
        """Stochastic Universal Sampling."""
        n_particles = len(self.particles)
        new_particles = []
        cumulative_sum = np.cumsum(self.weights)
        r = np.random.random() / n_particles
        for i in range(n_particles):
            u = r + i / n_particles
            idx = 0
            while idx < n_particles - 1 and u > cumulative_sum[idx]:
                idx += 1
            new_particle = self.particles[idx].copy()
            new_particle.x += np.random.normal(0, 0.01)
            new_particle.y += np.random.normal(0, 0.01)
            new_particle.theta += np.random.normal(0, 0.01)
            new_particles.append(new_particle)
        self.particles = new_particles
        self.weights = np.ones(n_particles) / n_particles

    def handle_odometry(self, pose_msg):
        """Compute relative motion and update camera-to-world transform."""
        self.last_robot_odom = self.robot_odom
        self.robot_odom = pose_msg
        if self.last_robot_odom:
            p_curr = np.array([
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z
            ])
            p_last = np.array([
                self.last_robot_odom.pose.pose.position.x,
                self.last_robot_odom.pose.pose.position.y,
                self.last_robot_odom.pose.pose.position.z
            ])
            q_last = np.array([
                self.last_robot_odom.pose.pose.orientation.x,
                self.last_robot_odom.pose.pose.orientation.y,
                self.last_robot_odom.pose.pose.orientation.z,
                self.last_robot_odom.pose.pose.orientation.w
            ])
            q_curr = np.array([
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w
            ])
            R_last = tr.quaternion_matrix(q_last)[0:3, 0:3]
            p_diff = R_last.transpose() @ (p_curr - p_last)
            q_diff = tr.quaternion_multiply(tr.quaternion_inverse(q_last), q_curr)
            _, _, yaw_diff = tr.euler_from_quaternion(q_diff)
            dt = (pose_msg.header.stamp.sec - self.last_robot_odom.header.stamp.sec) + \
                 (pose_msg.header.stamp.nanosec - self.last_robot_odom.header.stamp.nanosec) / 1e9
            self.dx = p_diff[0] / dt if dt > 0 else 0
            self.dy = p_diff[1] / dt if dt > 0 else 0
            self.dyaw = yaw_diff / dt if dt > 0 else 0
        # Update camera-to-world transform
        mean_pose, _ = self.compute_pose_mean_covariance()
        R = tr.quaternion_matrix([
            mean_pose.orientation.x,
            mean_pose.orientation.y,
            mean_pose.orientation.z,
            mean_pose.orientation.w
        ])[0:3, 0:3]
        t = np.array([
            mean_pose.position.x,
            mean_pose.position.y,
            mean_pose.position.z
        ])
        self.camera_to_world = (R, t)

    def handle_objects(self, objects_msg):
        """Process detected objects and update object dictionary."""
        if self.camera_to_world is None:
            return
        R_world_to_camera = self.camera_to_world[0].transpose()
        t_world_to_camera = -R_world_to_camera @ self.camera_to_world[1]
        for obj in objects_msg.objects:
            if obj.label not in ['person', 'vehicle']:
                continue
            pos_world = np.array([
                obj.position[0],
                obj.position[1],
                obj.position[2]
            ])
            # Apply distance threshold
            dist = sqrt(sum(pos_world**2))
            if dist < self.min_distance or dist > self.max_distance:
                continue
            # Convert to camera frame
            pos_camera = R_world_to_camera @ pos_world + t_world_to_camera
            # Convert back to world frame using particle filter's pose
            pos_world_corrected = self.camera_to_world[0] @ pos_camera + self.camera_to_world[1]
            # Compute variance (trace of covariance)
            cov = np.array(obj.position_covariance).reshape(3, 3) if len(obj.position_covariance) == 9 else np.diag(obj.position_covariance[:3])
            variance = np.trace(cov)
            obj_id = obj.label_id
            # Update dictionary (no removal)
            if obj_id not in self.object_dict or variance < self.object_dict[obj_id]['variance']:
                self.object_dict[obj_id] = {
                    'class': obj.label,
                    'position': pos_world_corrected.tolist(),
                    'variance': variance
                }

    def predict_particle_odometry(self, particle, dt):
        """Update particle pose."""
        if abs(self.dx) < 1e-10 and abs(self.dy) < 1e-10 and abs(self.dyaw) < 1e-5:
            return
        v = sqrt(self.dx**2 + self.dy**2)
        nx = np.random.normal(0, self.dynamics_translation_noise_std_dev)
        ny = np.random.normal(0, self.dynamics_translation_noise_std_dev)
        ntheta = np.random.normal(0, self.dynamics_orientation_noise_std_dev)
        particle.x += v * cos(particle.theta) * dt + nx
        particle.y += v * sin(particle.theta) * dt + ny
        particle.theta += self.dyaw * dt + ntheta

    def save_objects_to_csv(self, filename='objects.csv'):
        """Serialize object dictionary to CSV."""
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

    def compute_pose_mean_covariance(self):
        """Compute mean and covariance of robot pose from particles."""
        poses = np.array([[p.x, p.y, p.theta] for p in self.particles])
        weights = np.array(self.weights)
        mean_pose = np.average(poses, axis=0, weights=weights)
        diff = poses - mean_pose
        cov = np.cov(diff.T, aweights=weights, ddof=1)

        # Craft message to be sent for debugging/visualization in RViz
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose.position.x = mean_pose[0]
        pose_msg.pose.pose.position.y = mean_pose[1]
        pose_msg.pose.pose.position.z = 0.0
        q = tr.quaternion_from_euler(0, 0, mean_pose[2])
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]
        pose_msg.pose.covariance = np.zeros(36)

        # Covariance matrix in 6x6 is assumed to have indepedence between states (x, y, z, theta)
        # This way we only need to set the variance
        # The pose covariance is calculated from particles covariance 
        # since its calculated via the mean and variance of particles that guestimate where robot is
        pose_msg.pose.covariance[0] = cov[0, 0]  # x-x
        pose_msg.pose.covariance[1] = cov[0, 1]  # x-y
        pose_msg.pose.covariance[5] = cov[1, 1]  # y-y
        pose_msg.pose.covariance[6] = cov[1, 0]  # y-x
        pose_msg.pose.covariance[35] = cov[2, 2]  # theta-theta
        return pose_msg.pose, cov

    @staticmethod
    def metric_to_grid_coords(x, y, ogm):
        """Convert metric coordinates to OGM grid coordinates."""
        origin_x = ogm.info.origin.position.x
        origin_y = ogm.info.origin.position.y
        resolution = ogm.info.resolution
        height = ogm.info.height
        width = ogm.info.width
        gx = (x - origin_x) / resolution
        gy = (y - origin_y) / resolution
        row = min(max(int(gy), 0), height-1)
        col = min(max(int(gx), 0), width-1)
        return row, col

class AMCLPointCloud(Node):
    def __init__(self, num_particles, xmin, xmax, ymin, ymax):
        super().__init__('amcl_point_cloud')
        self.declare_parameter('map_file', '')
        self.declare_parameter('dynamics_translation_noise_std_dev', 0.1)
        self.declare_parameter('dynamics_orientation_noise_std_dev', 0.05)
        self.declare_parameter('point_cloud_measurement_noise_std_dev', 0.1)
        self.declare_parameter('min_distance', 0.5)
        self.declare_parameter('max_distance', 10.0)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 20.0)
        map_file = self.get_parameter('map_file').value

        # params that cannot be obtained from any topics or file - its just based on some prior knowledge
        # These will be used to propragate std dev for variance calculations later... so we preset them as some constant
        dynamics_translation_noise_std_dev = self.get_parameter('dynamics_translation_noise_std_dev').value
        dynamics_orientation_noise_std_dev = self.get_parameter('dynamics_orientation_noise_std_dev').value
        point_cloud_measurement_noise_std_dev = self.get_parameter('point_cloud_measurement_noise_std_dev').value

        # Load in the point cloud
        with open(map_file, 'rb') as f:
            map_point_cloud = pickle.load(f)
        map_points = []
        for p in point_cloud2.read_points(map_point_cloud, field_names=("x", "y", "z")):
            map_points.append([p[0], p[1], p[2]])
        map_points = np.array(map_points)


        self.ogm = None
        self.pf = ParticleFilter(
            num_particles, map_points, self.ogm, xmin, xmax, ymin, ymax,
            dynamics_translation_noise_std_dev,
            dynamics_orientation_noise_std_dev,
            point_cloud_measurement_noise_std_dev, self, n_thresh=num_particles / 2
        )
        self.pf.init_particles()
        self.last_point_cloud = None
        self.mutex = Lock()
        self.camera_info_received = False
        self.particles_pub = self.create_publisher(MarkerArray, 'particle_filter/particles', 1)
        self.debug_point_cloud_pub = self.create_publisher(PointCloud2, 'debug/point_cloud', 1)
        self.objects_point_cloud_pub = self.create_publisher(PointCloud2, 'objects_point_cloud', 1)
        self.amcl_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 1)
        self.odom_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/zed/zed_node/pose', self.odometry_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.point_cloud_sub = self.create_subscription(
            PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.point_cloud_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.ogm_sub = self.create_subscription(
            OccupancyGrid, '/rtabmap/grid_map', self.ogm_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.objects_sub = self.create_subscription(
            ObjectsStamped, '/zed/zed_node/obj_det/objects', self.objects_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/zed/zed_node/rgb/camera_info', self.camera_info_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.timer = self.create_timer(1.0, self.publish_particle_markers)
        # Timer to check for missing camera info
        self.camera_info_timeout = self.create_timer(5.0, self.check_camera_info)

    def check_camera_info(self):
        """Log error if camera info is missing."""
        if not self.camera_info_received:
            self.get_logger().error("No valid CameraInfo received on /zed/zed_node/rgb/camera_info. Using default parameters, which may degrade frustum simulation.")
        self.camera_info_timeout.cancel()

    def camera_info_callback(self, msg):
        """Update camera intrinsic parameters."""
        if msg.k[0] <= 0 or msg.k[4] <= 0 or msg.width <= 0 or msg.height <= 0:
            self.get_logger().error("Invalid CameraInfo: focal lengths or image dimensions are invalid. Using default parameters.")
            return
        self.pf.fx = msg.k[0]  # focal_length_x
        self.pf.fy = msg.k[4]  # focal_length_y
        self.pf.cx = msg.k[2]  # principal_point_x
        self.pf.cy = msg.k[5]  # principal_point_y
        self.pf.img_width = msg.width
        self.pf.img_height = msg.height
        self.camera_info_received = True
        self.get_logger().info("CameraInfo received and parameters updated.")

    def ogm_callback(self, msg):
        """Set static OGM (loaded once)."""
        if self.pf.ogm is None:
            self.mutex.acquire()
            self.pf.ogm = msg
            self.pf.grid_map = np.array(msg.data, dtype='int8').reshape(msg.info.height, msg.info.width)
            self.pf.grid_bin = (self.pf.grid_map == 0).astype('uint8')
            filtered_points = []
            for p in self.pf.map_points:
                row, col = self.pf.metric_to_grid_coords(p[0], p[1], msg)
                if row < msg.info.height and col < msg.info.width and self.pf.grid_bin[row, col]:
                    filtered_points.append(p)
            self.pf.map_points = np.array(filtered_points) if filtered_points else self.pf.map_points
            self.pf.map_kdtree = KDTree(self.pf.map_points)
            self.mutex.release()

    def odometry_callback(self, msg):
        self.mutex.acquire()
        self.pf.handle_odometry(msg)
        self.mutex.release()

    def point_cloud_callback(self, msg):
        dt = 0
        if self.last_point_cloud:
            dt = (msg.header.stamp.sec - self.last_point_cloud.header.stamp.sec) + \
                 (msg.header.stamp.nanosec - self.last_point_cloud.header.stamp.nanosec) / 1e9
        self.mutex.acquire()
        self.publish_debug_point_cloud(msg)
        self.pf.handle_observation(msg, dt)
        self.pf.dx = 0
        self.pf.dy = 0
        self.pf.dyaw = 0
        self.mutex.release()
        self.last_point_cloud = msg

    def objects_callback(self, msg):
        self.mutex.acquire()
        self.pf.handle_objects(msg)
        self.publish_objects_point_cloud()
        self.mutex.release()

    def publish_debug_point_cloud(self, msg):
        points, _ = self.pf.subsample_point_cloud(msg)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='zed2_camera')
        fields = [
            {'name': 'x', 'offset': 0, 'datatype': 7, 'count': 1},
            {'name': 'y', 'offset': 4, 'datatype': 7, 'count': 1},
            {'name': 'z', 'offset': 8, 'datatype': 7, 'count': 1}
        ]
        cloud_msg = point_cloud2.create_cloud(header, fields, points)
        self.debug_point_cloud_pub.publish(cloud_msg)

    def publish_objects_point_cloud(self):
        points = [data['position'] for data in self.pf.object_dict.values()]
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        fields = [
            {'name': 'x', 'offset': 0, 'datatype': 7, 'count': 1},
            {'name': 'y', 'offset': 4, 'datatype': 7, 'count': 1},
            {'name': 'z', 'offset': 8, 'datatype': 7, 'count': 1}
        ]
        cloud_msg = point_cloud2.create_cloud(header, fields, points)
        self.objects_point_cloud_pub.publish(cloud_msg)

    def publish_particle_markers(self):
        marker_array = MarkerArray()
        for i, particle in enumerate(self.pf.particles):
            marker = self.get_particle_marker(particle, i)
            marker_array.markers.append(marker)
        pos = np.array([[p.x, p.y, p.theta] for p in self.pf.particles])
        avg_pos = np.average(pos, axis=0, weights=self.pf.weights)
        avg_particle = Particle(-1, avg_pos[0], avg_pos[1], avg_pos[2])
        avg_marker = self.get_particle_marker(avg_particle, len(self.pf.particles), avg=True)
        marker_array.markers.append(avg_marker)
        self.particles_pub.publish(marker_array)
        pose_msg, _ = self.pf.compute_pose_mean_covariance()
        self.amcl_pose_pub.publish(pose_msg)

    def get_particle_marker(self, particle, marker_id, avg=False):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'particles'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.lifetime.sec = 1
        yaw = particle.theta
        vx = cos(yaw)
        vy = sin(yaw)
        if avg:
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            marker.points.append(Point(x=particle.x, y=particle.y, z=0.3))
            marker.points.append(Point(x=particle.x + 0.3*vx, y=particle.y + 0.3*vy, z=0.3))
        else:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.1)
            marker.points.append(Point(x=particle.x, y=particle.y, z=0.2))
            marker.points.append(Point(x=particle.x + 0.3*vx, y=particle.y + 0.3*vy, z=0.2))
        marker.scale.x = 0.05
        marker.scale.y = 0.15
        marker.scale.z = 0.1
        return marker

    def shutdown(self):
        self.pf.save_objects_to_csv()
        super().destroy_node()

def main(args=None):
    # Find out by opening RTabMapViz
    xmin = -10.0
    xmax = 10.0
    ymin = -10.0
    ymax = 10.0

    # I swear the Jetson will explode if you set this to 2000
    num_particles = 100
    rclpy.init(args=args)
    node = AMCLPointCloud(num_particles, xmin, xmax, ymin, ymax)
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
