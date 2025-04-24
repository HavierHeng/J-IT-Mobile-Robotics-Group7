#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf_transformations as tr
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from zed_interfaces.msg import Object, ObjectsStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Float32
from math import sqrt
import numpy as np
import csv
from threading import Lock
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

class ObjectTrackerRTabMapSLAM(Node):
    def __init__(self):
        super().__init__('object_tracker_rtabmap_slam')

        # Declare parameters
        self.declare_parameter('min_distance', 0.5)  # Min distance for object detection (m)
        self.declare_parameter('max_distance', 10.0)  # Max distance for object detection (m)
        self.declare_parameter('cluster_distance', 1.0)  # Distance threshold for clustering (m)
        self.declare_parameter('map_update_interval', 5.0)  # Seconds between KDTree updates
        self.declare_parameter('map_covariance_std', 0.1)  # Map point std dev (m)

        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.cluster_distance = self.get_parameter('cluster_distance').value
        self.map_update_interval = self.get_parameter('map_update_interval').value
        self.map_covariance_std = self.get_parameter('map_covariance_std').value

        # Object dictionary and state
        self.object_dict = {}
        self.object_id_counter = 0
        self.mutex = Lock()

        # Robot pose and map state
        self.robot_pose = None
        self.robot_cov = None
        self.pose_received = False
        self.map_points = None
        self.map_kdtree = None
        self.last_map_points = None

        # Transformation matrices
        self.R_map_camera = None
        self.t_map_camera = None
        self.R_odom_camera = None
        self.t_odom_camera = None

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
        self.map_sub = self.create_subscription(
            PointCloud2,
            '/rtabmap/cloud_map',
            self.map_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publishers for debugging
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.shadow_pub = self.create_publisher(Marker, '/debug/object_shadow', qos_profile)
        self.distance_pub = self.create_publisher(Float32, '/debug/distance_to_object', qos_profile)

        # Timer for pose timeout
        self.pose_timeout = self.create_timer(5.0, self.check_pose)

        # Logging for ground truth comparison
        self.log_file = open('objects_slam.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['Timestamp', 'Object_ID', 'Class', 'X', 'Y', 'Z', 'Variance', 'Robot_X', 'Robot_Y', 'Robot_Theta'])

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

    def map_callback(self, msg):
        self.mutex.acquire()
        try:
            points = []
            for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                points.append([p['x'], p['y'], p['z']])
            self.map_points = np.array(points)
            if len(self.map_points) > 0:
                self.map_kdtree = cKDTree(self.map_points)
            else:
                self.get_logger().warn("Empty point cloud received")
                return

            # Detect map update (e.g., loop closure)
            if self.last_map_points is not None and len(self.map_points) > len(self.last_map_points) * 1.1:
                self.update_object_positions()
                objects = [
                    {'class': data['class'], 'position': data['position'], 'variance': data['variance']}
                    for data in self.object_dict.values()
                ]
                self.object_dict.clear()
                self.object_id_counter = 0
                self.cluster_and_update_objects(objects)
            self.last_map_points = self.map_points.copy()
        finally:
            self.mutex.release()

    def objects_callback(self, msg):
        self.mutex.acquire()
        try:
            if self.robot_pose is None or self.R_odom_camera is None or self.map_kdtree is None:
                self.get_logger().warn("Missing robot pose, odometry, or map")
                return

            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            new_objects = []
            for obj in msg.objects:
                if obj.label not in ['person', 'vehicle']:
                    continue
                pos_odom = np.array([obj.position[0], obj.position[1], obj.position[2]])
                R_odom_to_camera = self.R_odom_camera.transpose()
                t_odom_to_camera = -R_odom_to_camera @ self.t_odom_camera
                pos_camera = R_odom_to_camera @ pos_odom + t_odom_to_camera
                dist = sqrt(sum(pos_camera**2))
                if dist < self.min_distance or dist > self.max_distance:
                    continue
                pos_map = self.R_map_camera @ pos_camera + self.t_map_camera
                dist, _ = self.map_kdtree.query(pos_map)
                if dist > 0.5:
                    continue
                obj_cov = np.array(obj.position_covariance).reshape(3, 3) if len(obj.position_covariance) == 9 else np.diag(obj.position_covariance[:3])
                robot_cov_2d = np.array([
                    [self.robot_cov[0, 0], self.robot_cov[0, 1], 0],
                    [self.robot_cov[1, 0], self.robot_cov[1, 1], 0],
                    [0, 0, 0]
                ])
                map_cov = np.diag([self.map_covariance_std**2] * 3)
                combined_cov = obj_cov + self.R_map_camera @ robot_cov_2d @ self.R_map_camera.T + map_cov
                variance = np.trace(combined_cov)
                new_objects.append({
                    'class': obj.label,
                    'position': pos_map.tolist(),
                    'variance': variance,
                    'raw_obj': obj
                })
                self.publish_shadow(pos_map)
                # Log for ground truth comparison
                robot_theta = tr.euler_from_quaternion([
                    self.robot_pose.orientation.x,
                    self.robot_pose.orientation.y,
                    self.robot_pose.orientation.z,
                    self.robot_pose.orientation.w
                ])[2]
                self.csv_writer.writerow([
                    timestamp,
                    self.object_id_counter,  # Temporary ID for logging
                    obj.label,
                    pos_map[0],
                    pos_map[1],
                    pos_map[2],
                    variance,
                    self.t_map_camera[0],
                    self.t_map_camera[1],
                    robot_theta
                ])

            if new_objects:
                self.cluster_and_update_objects(new_objects)
                nearest_obj = min(new_objects, key=lambda o: sqrt(sum((np.array(o['position']) - self.t_map_camera)**2)))
                distance = sqrt(sum((np.array(nearest_obj['position'][:2]) - self.t_map_camera[:2])**2))
                distance_msg = Float32(data=float(distance))
                self.distance_pub.publish(distance_msg)
        finally:
            self.mutex.release()
            self.log_file.flush()

    def cluster_and_update_objects(self, new_objects):
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

    def update_object_positions(self):
        self.mutex.acquire()
        try:
            for obj_id, data in self.object_dict.items():
                pos = np.array(data['position'])
                # Simplified shift (in practice, use RTabMap's graph optimization)
                data['position'] = (pos + np.random.normal(0, 0.05, 3)).tolist()
                data['variance'] += 0.01
        finally:
            self.mutex.release()

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

    def save_objects_to_csv(self, filename='objects_slam.csv'):
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
        self.log_file.close()
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerRTabMapSLAM()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()