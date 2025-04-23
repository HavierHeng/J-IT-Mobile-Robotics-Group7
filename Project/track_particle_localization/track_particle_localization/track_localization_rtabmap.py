#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf_transformations as tr
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from zed_interfaces.msg import Object, ObjectsStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Float32
from math import sqrt
import numpy as np
import csv
from threading import Lock
from sklearn.cluster import DBSCAN

class ObjectTrackerRTabMap(Node):
    def __init__(self):
        super().__init__('object_tracker_rtabmap')

        # Declare parameters
        self.declare_parameter('min_distance', 0.5)  # Min distance for object detection (m)
        self.declare_parameter('max_distance', 10.0)  # Max distance for object detection (m)
        self.declare_parameter('cluster_distance', 1.0)  # Distance threshold for clustering objects (m)

        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.cluster_distance = self.get_parameter('cluster_distance').value

        # Object dictionary to store detections
        self.object_dict = {}
        self.object_id_counter = 0  # For assigning unique IDs
        self.mutex = Lock()

        # Robot pose from RTabMap
        self.robot_pose = None
        self.robot_cov = None
        self.pose_received = False

        # Transformation matrices
        self.R_map_camera = None  # Rotation from camera to map frame
        self.t_map_camera = None  # Translation from camera to map frame
        self.R_odom_camera = None  # Rotation from camera to odom frame
        self.t_odom_camera = None  # Translation from camera to odom frame

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

        # Publishers for debugging
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.shadow_pub = self.create_publisher(Marker, '/debug/object_shadow', qos_profile)
        self.distance_pub = self.create_publisher(Float32, '/debug/distance_to_object', qos_profile)

        # Timer to check for missing pose
        self.pose_timeout = self.create_timer(5.0, self.check_pose)

    def check_pose(self):
        """Log error if no valid pose is received."""
        if not self.pose_received:
            self.get_logger().error("No valid PoseWithCovarianceStamped received on /rtabmap/localization_pose. Object tracking disabled.")
        self.pose_timeout.cancel()

    def pose_callback(self, msg):
        """Update robot pose and covariance from RTabMap."""
        self.mutex.acquire()
        try:
            # Validate covariance
            cov = np.array(msg.pose.covariance).reshape(6, 6)
            if np.any(np.diag(cov) < 0):
                self.get_logger().warn("Invalid covariance in /rtabmap/localization_pose. Ignoring message.")
                return
            self.robot_pose = msg.pose.pose
            self.robot_cov = cov
            self.pose_received = True

            # Update transformation matrices
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
        """Update odometry transformation for world-to-camera conversion."""
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

    def objects_callback(self, msg):
        """Process detected objects, transform positions, and update dictionary."""
        self.mutex.acquire()
        try:
            if self.robot_pose is None or self.R_odom_camera is None:
                self.get_logger().warn("Missing robot pose or odometry. Skipping object processing.")
                return

            # Process each object
            new_objects = []
            for obj in msg.objects:
                if obj.label not in ['person', 'vehicle']:
                    continue

                # Object position in world (odom) frame
                pos_odom = np.array([
                    obj.position[0],
                    obj.position[1],
                    obj.position[2]
                ])

                # Transform to camera frame
                R_odom_to_camera = self.R_odom_camera.transpose()
                t_odom_to_camera = -R_odom_to_camera @ self.t_odom_camera
                pos_camera = R_odom_to_camera @ pos_odom + t_odom_to_camera

                # Check distance in camera frame
                dist = sqrt(sum(pos_camera**2))
                if dist < self.min_distance or dist > self.max_distance:
                    continue

                # Transform to map frame
                pos_map = self.R_map_camera @ pos_camera + self.t_map_camera

                # Combine covariances
                obj_cov = np.array(obj.position_covariance).reshape(3, 3) if len(obj.position_covariance) == 9 else np.diag(obj.position_covariance[:3])
                robot_cov_2d = np.array([
                    [self.robot_cov[0, 0], self.robot_cov[0, 1], 0],
                    [self.robot_cov[1, 0], self.robot_cov[1, 1], 0],
                    [0, 0, 0]
                ])  # x, y covariance
                # Simplified: Transform robot covariance to object space and add
                combined_cov = obj_cov + self.R_map_camera @ robot_cov_2d @ self.R_map_camera.T
                variance = np.trace(combined_cov)

                # Store for clustering
                new_objects.append({
                    'class': obj.label,
                    'position': pos_map.tolist(),
                    'variance': variance,
                    'raw_obj': obj
                })

                # Publish debug shadow
                self.publish_shadow(pos_map)

            # Cluster objects to assign IDs
            if new_objects:
                self.cluster_and_update_objects(new_objects)

            # Publish distance to nearest object (for debugging)
            if new_objects:
                nearest_obj = min(new_objects, key=lambda o: sqrt(sum((np.array(o['position']) - self.t_map_camera)**2)))
                distance = sqrt(sum((np.array(nearest_obj['position'][:2]) - self.t_map_camera[:2])**2))
                distance_msg = Float32(data=float(distance))
                self.distance_pub.publish(distance_msg)

        finally:
            self.mutex.release()

    def cluster_and_update_objects(self, new_objects):
        """Cluster objects by position and update object dictionary."""
        # Extract positions for clustering
        positions = np.array([o['position'][:2] for o in new_objects])  # Use x, y only
        if len(positions) == 0:
            return

        # DBSCAN clustering
        clustering = DBSCAN(eps=self.cluster_distance, min_samples=1).fit(positions)
        labels = clustering.labels_

        # Process each cluster
        for label in set(labels):
            if label == -1:  # Noise points
                continue
            cluster_objs = [o for i, o in enumerate(new_objects) if labels[i] == label]
            cluster_positions = np.array([o['position'][:2] for o in cluster_objs])

            # Find closest existing object in dictionary
            best_match_id = None
            min_dist = float('inf')
            cluster_center = np.mean(cluster_positions, axis=0)
            for obj_id, data in self.object_dict.items():
                pos = np.array(data['position'][:2])
                dist = sqrt(sum((pos - cluster_center)**2))
                if dist < min_dist and data['class'] == cluster_objs[0]['class']:
                    min_dist = dist
                    best_match_id = obj_id

            # Select best object in cluster (lowest variance)
            best_obj = min(cluster_objs, key=lambda o: o['variance'])

            # Update or add to dictionary
            if best_match_id is not None and min_dist < self.cluster_distance:
                # Update existing object if new variance is lower
                if best_obj['variance'] < self.object_dict[best_match_id]['variance']:
                    self.object_dict[best_match_id] = {
                        'class': best_obj['class'],
                        'position': best_obj['position'],
                        'variance': best_obj['variance']
                    }
            else:
                # New object
                self.object_dict[self.object_id_counter] = {
                    'class': best_obj['class'],
                    'position': best_obj['position'],
                    'variance': best_obj['variance']
                }
                self.object_id_counter += 1

    def publish_shadow(self, pos_map):
        """Publish a shadow marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'objects'
        marker.id = int(str(id(pos_map))[-9:])  # Unique ID based on position
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
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerRTabMap()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()