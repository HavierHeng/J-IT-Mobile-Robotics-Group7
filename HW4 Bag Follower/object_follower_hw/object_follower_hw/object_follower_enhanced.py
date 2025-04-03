#!/usr/bin/env python3
import rclpy
import rclpy.duration
from rclpy.node import Node 
import tf_transformations as tr
import tf2_ros
from tf2_ros import TransformBroadcaster
from std_msgs.msg import String, Header, ColorRGBA, Float32
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from math import sqrt, cos, sin, pi, atan2, log, exp
from threading import Thread, Lock
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
import random
import numpy as np
import sys
from zed_interfaces.msg import Object, ObjectsStamped
from rclpy.serialization import deserialize_message
from typing import Optional

class ObjectFollower(Node):
    def __init__(self):
        super().__init__("ObjectFollower")

        self.declare_parameter('stopping_distance_to_object', 1.0)  # Distance for bag to set velocity to 0 (in metres)
        self.stopping_distance_to_object = self.get_parameter("stopping_distance_to_object").value

        # Additional parameters for following behavior
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('proportional_gain_linear', 0.5)
        self.declare_parameter('proportional_gain_angular', 1.0)
        
        self.max_linear_velocity = self.get_parameter("max_linear_velocity").value
        self.max_angular_velocity = self.get_parameter("max_angular_velocity").value
        self.proportional_gain_linear = self.get_parameter("proportional_gain_linear").value
        self.proportional_gain_angular = self.get_parameter("proportional_gain_angular").value

        # Subscription - For Zed2 detected objects
        self.obj_subscription = self.create_subscription(
                ObjectsStamped, 
                "/zed/zed_node/obj_det/objects",
                self.object_callback,
                10) 

        # Subscription - For Zed2 Odometry
        self.odometry = None
        self.odom_subscription = self.create_subscription(
                PoseStamped,
                "/zed/zed_node/pose",
                self.odom_callback,
                10)

        # Publish - For telling Car how to drive using Twist
        self.cmd_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Publish - For RViz - Shadow Marker for Bounding Box
        self.shadow_publisher = self.create_publisher(Marker, "/debug/object_shadow", 10)

        # Publish - For RViz - draw an arrow from shadow of zed_camera_center to shadow of bag bounding box
        self.camera_obj_arrow_publisher = self.create_publisher(Marker, "/debug/object_arrow", 10)

        # Publish - For Debug - Current distance to Object
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.distance_publisher = self.create_publisher(Float32, '/debug/distance_to_object', qos_profile)  # Float32 has data attribute

        # Odometry of camera to map - given by ~/pose
        self.q_map_cameracenter = None  # quaternion of camera center in map frame
        self.R_map_cameracenter = None  # 3x3 rotation matrix of camera center in map frame
        self.p_map_cameracenter = None  # position of camera center in map frame

    def odom_callback(self, msg):
        """
        PoseStamped from ~/pose from Zed2 to Pose message
        """
        self.odometry = msg.pose
        
    def object_callback(self, msg):
        """
        Main logic - Takes in ObjectStamped which contains multiple Object
        Calculates how much error between camera to bag and uses that to determine 
        linear and angular velocity to send to /cmd_vel
        """
        if self.odometry is None:
            self.get_logger().warn("No odometry data received yet")
            return

        # Get position of zed_camera_center in map frame coordinates based on current odometry message
        self.p_map_cameracenter = np.array([
            self.odometry.position.x,
            self.odometry.position.y,
            self.odometry.position.z
        ])

        # Get rotation of zed_camera_center to map frame
        self.q_map_cameracenter = np.array([
            self.odometry.orientation.x,
            self.odometry.orientation.y,
            self.odometry.orientation.z,
            self.odometry.orientation.w
        ])
        
        # Convert quaternion to rotation matrix
        # tr.quaternion_matrix returns a 4x4 homogeneous transformation matrix
        transform_matrix = tr.quaternion_matrix(self.q_map_cameracenter)
        self.R_map_cameracenter = transform_matrix[0:3, 0:3]  # Extract the 3x3 rotation matrix

        # Get nearest object if any objects are detected
        if len(msg.objects) == 0:
            self.get_logger().warn("No objects detected")
            # Stop the robot if no objects are detected
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_publisher.publish(cmd_msg)
            return
            
        nearest_obj = self._get_nearest_object(msg)
        if nearest_obj is None:
            return
        obj_corners = self._get_object_corners(nearest_obj)  # Get corners from object
        p_map_object = self._get_center_from_corners(obj_corners)  # Get center of object in map frame

        # Calculate distance to object (using only x,y for ground plane distance)
        distance_to_obj = np.linalg.norm(
            np.array([self.p_map_cameracenter[0], self.p_map_cameracenter[1]]) - 
            np.array([p_map_object[0], p_map_object[1]])
        )
        
        # Publish distance for debugging
        distance_msg = Float32()
        distance_msg.data = float(distance_to_obj)
        self.distance_publisher.publish(distance_msg)

        # Calculate error between desired stopping distance and current position
        linear_error = distance_to_obj - self.stopping_distance_to_object 

        # Calculate the angle to the object in the map frame
        # Vector from camera to object
        direction_vector = np.array([
            p_map_object[0] - self.p_map_cameracenter[0],
            p_map_object[1] - self.p_map_cameracenter[1]
        ])
        
        # Normalize the direction vector
        if np.linalg.norm(direction_vector) > 0:
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # The heading vector of the camera in the map frame (assuming forward is along x-axis in camera frame)
        heading_vector = self.R_map_cameracenter @ np.array([1.0, 0.0, 0.0])
        heading_vector = heading_vector[0:2]  # Take only x,y components
        
        # Normalize the heading vector
        if np.linalg.norm(heading_vector) > 0:
            heading_vector = heading_vector / np.linalg.norm(heading_vector)
        
        # Calculate the dot product and cross product to find the angle
        dot_product = np.dot(heading_vector, direction_vector)
        cross_product = np.cross(np.append(heading_vector, 0), np.append(direction_vector, 0))[2]
        
        # Calculate the angle between the vectors
        angle_to_object = atan2(cross_product, dot_product)
        
        # Control the robot with proportional control
        cmd_msg = Twist()
        
        # Linear velocity control (proportional to distance error)
        if linear_error > 0:
            # Proportional control for linear velocity
            cmd_msg.linear.x = min(self.proportional_gain_linear * linear_error, self.max_linear_velocity)
        else:
            cmd_msg.linear.x = 0.0
        
        # Angular velocity control (proportional to angular error)
        cmd_msg.angular.z = min(
            max(self.proportional_gain_angular * angle_to_object, -self.max_angular_velocity),
            self.max_angular_velocity
        )
        
        # Publish control command
        self.cmd_publisher.publish(cmd_msg)

        # For debugging - publish shadow of object
        shadow_marker = Marker()
        shadow_marker.header.frame_id = "map"  # Coordinate frame
        shadow_marker.header.stamp = self.get_clock().now().to_msg()
        shadow_marker.ns = "debug"
        shadow_marker.id = 0
        shadow_marker.type = Marker.SPHERE
        shadow_marker.action= Marker.ADD
        shadow_marker.scale.x = 0.1
        shadow_marker.scale.y = 0.1
        shadow_marker.scale.z = 0.1
        shadow_marker.color.a = 1.0
        shadow_marker.color.r = 0.0
        shadow_marker.color.g = 1.0
        shadow_marker.color.b = 0.0
        
        shadow_marker_point = Point() 
        shadow_marker_point.x = p_map_object[0]
        shadow_marker_point.y = p_map_object[1]
        shadow_marker_point.z = 0.0
        shadow_marker.points = [shadow_marker_point]
        self.shadow_publisher.publish(shadow_marker)

        # For debugging - Publish Arrow from camera_center shadow to object shadow
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"  # Coordinate frame
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.ns = "debug"
        arrow_marker.id = 1
        arrow_marker.type = Marker.ARROW
        arrow_marker.action= Marker.ADD
        arrow_marker.scale.x = 0.1
        arrow_marker.scale.y = 0.1
        start_point = Point()
        start_point.x = self.p_map_cameracenter[0]
        start_point.y = self.p_map_cameracenter[1]
        start_point.z = 0.0
        end_point = Point()
        end_point.x = p_map_object[0]
        end_point.y = p_map_object[1]
        end_point.z = 0.0
        arrow_marker.points = [start_point, end_point]
        arrow_marker.color.a = 1.0
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 1.0

    def _get_nearest_object(self, msg: ObjectsStamped) -> Optional[Object]:
        """
        Given a msg containing ObjectsStamped, pick out the Object with the minimum distance from current position to object.
        """
        if len(msg.objects) == 0:
            return None

        objects = np.array([[obj.position[0], 
                             obj.position[1], 
                             obj.position[2]] for obj in msg.objects])

        # Calculate L2 Distance from camera center to each object
        distances = np.linalg.norm(objects - self.p_map_cameracenter, axis=1)

        # Find index of closest object
        k = np.argmin(distances)
        return msg.objects[k]
        
    def _get_object_corners(self, msg: Object):
        """
        Objects which contain BoundingBox3D, which contains Keypoint3D (8 corners), contains 3 Floats
        """
        corners_obj = msg.bounding_box_3d.corners  # 8 Corners in an array
        corners = [[c.kp[0], c.kp[1], c.kp[2]] for c in corners_obj]  # Access Float values
        return corners

    def _get_center_from_corners(self, vertices):
        """
        Corners is a list of [x, y, z] points making up the cube corners.
        Calculate the center of the bounding box.
        """
        points = np.array(vertices)
        # Calculate the mean along axis 0 (across all points)
        center = np.mean(points, axis=0)
        return center.tolist()

    def run(self):
        """
        For running the node as a standalone. In practice, the node is spun by rclpy.spin(node).
        """
        rate = self.create_rate(200)
        while rclpy.ok():
            rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    m = ObjectFollower()
    try:
        rclpy.spin(m)
    except KeyboardInterrupt:
        pass
    finally:
        m.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
