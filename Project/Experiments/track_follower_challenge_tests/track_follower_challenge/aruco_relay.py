#!/usr/bin/env python3
import rclpy
import rclpy.duration
from rclpy.node import Node 
import tf_transformations as tr
import tf2_ros
from tf2_ros import TransformBroadcaster
from std_msgs.msg import String, Header, ColorRGBA, Float32
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point, PoseArray
from sensor_msgs.msg import Image, CameraInfo
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
from typing import Optional


class AruCoRelay(Node):
    def __init__(self):
        super().__init__("AruCoRelay")

        # QoS settings to match ZED topics (often use sensor data profile)
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', qos_profile)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', qos_profile)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb_raw/image_raw_color',
            self.image_callback,
            qos_profile
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/rgb_raw/camera_info',
            self.camera_info_callback,
            qos_profile
        )

        self.aruco_poses_sub = self.create_subscription(
            PoseArray,
            '/aruco_poses',
            self.aruco_poses_callback,
            qos_profile
        )

        self.get_logger().info("AruCoRelay node started, relaying ZED camera image and camera_info topics.")


    def image_callback(self, msg):
        """
        Subscribes: ~/zed/zed_node/rgb_raw/image_raw_color
        Publishes: ~/camera/image_raw
        Type: sensor_msgs.msg.Image
        """
        self.image_pub.publish(msg)

    def camera_info_callback(self, msg):
        """
        Subscribes: ~/zed/zed_node/rgb_raw/camera_info
        Publishes: ~/camera/camera_info
        Type: sensor_msgs.msg.CameraInfo
        """
        self.camera_info_pub.publish(msg)

    def aruco_poses_callback(self, msg: PoseArray):
        marker_count = len(msg.poses)
        self.get_logger().info(f"[AruCoRelay] Detected {marker_count} marker pose(s) via /aruco_poses")

def main(args=None):
    rclpy.init(args=args)
    node = AruCoRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()