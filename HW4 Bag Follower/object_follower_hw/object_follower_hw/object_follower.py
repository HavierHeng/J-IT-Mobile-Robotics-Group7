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
from math import sqrt, cos, sin, pi, atan2, log ,exp
from threading import Thread, Lock
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
import random
import numpy as np
import sys
# from rosbag2_py_ import SequentialReader, StorageOptions, ConverterOptions
from zed_interfaces.msg import Object, ObjectsStamped
from rclpy.serialization import deserialize_message

class ObjectFollower(Node):
    def __init__(self):
        super().__init__("ObjectFollower")

        self.declare_parameter('stopping_distance_to_object', 1.0)  # Distance for bag to set velocity to 0 (in metres)
        self.stopping_distance_to_object = self.get_parameter("stopping_distance_to_object").value

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

        # Zed2 is able to get odom from ~/pose - it implicitly does EKF with IMU + visual odom
        # self.tf_buffer = tf2_ros.Buffer()  # Use Buffer for TransformListener
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Odometry of camera to map - given by ~/pose
        self.q_map_cameracenter = None  # quaternion of basefootprint in map frame
        self.R_map_cameracenter = None  # 3x3 rotation matrix of basefootprint in map frame
        self.p_map_cameracenter = None  # position of basefootprint in map frame

        # Object Tracking of camera (zed_camera_center) to object
        # self.q_cameracenter_object = None  # quaternion of basescan in map frame
        # self.R_cameracenter_object = None  # 3x3 rotation matrix of basescan in map frame
        # self.p_cameracenter_object = None  # position of basescan in map frame


    def odom_callback(self, msg):
        """
        PoseStamped from ~/pose from Zed2 to Pose message
        """
        self.odometry = msg.pose
        
    def object_callback(self, msg):
        """
        Main logic - Takes in ObjectStamped which contains multiple Object
        Calculates how much error between camera to bag and uses that to determine linear velocity to send to /cmd_vel
            - If less than desired stopping distance, then velocity is set to 0
            - If more than desired stopping distance, then velocity is 1.0 (Might want to explore if we are following object of arbitrary orientation)
        """
        # Get Odometry from zed_camera_center to map frame
        # Get position of zed_camera_center in map frame coordinates based on current odometry mmessage
        self.p_map_cameracenter = np.array([self.odometry.pose.pose.position.x,
                                            self.odometry.pose.pose.position.y,
                                            self.odometry.pose.pose.position.z
                                            ])

        # TODO: Get rotation of zed_camera_center to map frame - this is for following an object not directly in front
        # Task : Get the quaternion of the base_footprint frame in the map frame based on the current odometry message. 
        # Hints: Functions you can use from transformation
        # self.q_map_cameracenter = np.array([self.odometry.pose.pose.orientation.x,
        #                                     self.odometry.pose.pose.orientation.y,
        #                                     self.odometry.pose.pose.orientation.z,
        #                                     self.odometry.pose.pose.orientation.w
        #                                     ])
        # tr.quaternion_matrix converts form quaternion to rotation matrix
        # self.R_map_cameracenter = tr.quaternion_matrix()[0:3, 0:3]
        # self.q_map_basescan = tr.quaternion_multiply()
        # Note: A@B performs matrix multiplication between numpy matrices A and B, whereas A*B is element-wise multiplication
        # self.p_map_basescan = self.p_map_basefootprint + self.R_map_basefootprint @ self.p_basefootprint_basescan

        # Get Object bounding box(es) from ObjectsStamped in msg - it contains a list of Object in map frame
        # Note: If there are more than one Object, take pick closest distance object, this way no false positive (even for race track challenge)
        nearest_obj = self._get_nearest_object(msg)
        obj_corners = self._get_object_corners(nearest_obj)  # Get corners from object
        p_map_object = self._get_center_from_corners(obj_corners)  # Get center of object in map frame

        # Take shadow (x, y only) of camera_center and object and calculate distance
        # Else it be in 3D and subject to the height that the object is seen
        distance_to_obj = np.linalg.norm(self.p_map_cameracenter[0:1] - p_map_object[0:1])

        # Calculate error between desired stopping distance and current position
        error = distance_to_obj - self.stopping_distance_to_object 

        # Control car (Bang bang control)
        cmd_msg = Twist()
        if error > 0:  # Set linear x to 1.0
            cmd_msg.linear.x = 1.0
        else:  # Set linear x to zero so it stops moving
            cmd_msg.linear.x = 0.0

        # Publish control
        self.cmd_publisher.publish(cmd_msg)

        # For debugging - publish shadow of object
        shadow_marker = Marker()
        shadow_marker.header.frame_id = "map"  # Coordinate frame
        shadow_marker.header.stamp = self.get_clock().now().to_msg()
        shadow_marker.ns = "debug"
        shadow_marker.id = 0
        shadow_marker.marker_type = Marker.SPHERE
        shadow_marker.action= Marker.ADD
        shadow_marker.scale.x = 0.1
        shadow_marker.scale.y = 0.1
        shadow_marker.scale.z = 0.1
        shadow_marker.color.a = 1.0
        shadow_marker.color.r = 0.0
        shadow_marker.color.g = 1.0
        shadow_marker.color.b = 0.0
        self.shadow_publisher.publish(shadow_marker)

        # For debugging - Publish Arrow from camera_center shadow to object shadow
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"  # Coordinate frame
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.ns = "debug"
        arrow_marker.id = 1
        arrow_marker.marker_type = Marker.ARROW
        arrow_marker.action= Marker.ADD
        start_point = Point()
        start_point.x = self.p_map_cameracenter[0]
        start_point.y = self.p_map_cameracenter[1]
        start_point.z = 0
        end_point = Point()
        end_point.x = p_map_object[0]
        end_point.y = p_map_object[1]
        end_point.z = 0
        arrow_marker.points = [start_point, end_point]
        arrow_marker.color.a = 1.0
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 1.0

        self.camera_obj_arrow_publisher.publish(arrow_marker)

    def _get_nearest_object(self, msg: ObjectsStamped) -> Object:
        """
        Given a msg containing ObjectsStamped, pick out the Object with the minimum distance from current position to object.
        """
        objects = np.array([[obj.position[0].items(), 
                             obj.position[1].items(), 
                             obj.position[2].items()] for obj in msg.objects])

        # Calculate L2 Distance from camera center to each object
        distances = np.linalg.norm(objects - self.p_map_cameracenter, axis=1)

        # Find index of closest object
        k = np.argmin(distances)
        return msg.objects[k]
        

    def _get_object_shadow_marker(self, timestamp, frame_id, pts_in_map):
        """
        Plot shadow of the middle of 3D Bounding Box
        """
        msg = Marker()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.ns = "shadow_pts"
        msg.id = 0
        msg.type = Marker.POINTS
        msg.action = Marker.ADD
        msg.points = [Point(x=pt[0], y=pt[1], z=pt[2]) for pt in pts_in_map]
        msg.colors = [ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) for _ in pts_in_map]
        
        for pt in pts_in_map:
            assert ((not np.isnan(pt).any()) and np.isfinite(pt).all())
        
        msg.scale.x = 0.04
        msg.scale.y = 0.04
        msg.scale.z = 0.04
        return msg

    def _get_object_corners(self, msg: Object):
        """
        Objects which contain BoundingBox3D, which contains Keypoint3D (8 corners), contains 3 Floats
        """
        corners_obj = msg.bounding_box_3d.corners  # 8 Corners in an array
        corners = [[c.kp[0].item(), c.kp[1].item(), c.kp[2].item()] for c in corners_obj]  # item() is since its ROS Float
        return corners

    def _get_center_from_corners(self, vertices):
        """
        Corners is a list of [x, y  z] points making up the cube corners.
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
    

