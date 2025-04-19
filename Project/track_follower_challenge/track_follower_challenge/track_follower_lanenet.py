#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from lanenet_model import lanenet
from lanenet_postprocess import LaneNetPostProcessor

class PID:
    def __init__(self, Kp, Kd, Ki, dt=0.01):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0
        self.correction = 0.0

    def update_control(self, current_error):
        self.integral += current_error * self.dt
        derivative = (current_error - self.prev_error) / self.dt
        self.correction = self.Kp * current_error + self.Kd * derivative + self.Ki * self.integral
        self.prev_error = current_error

    def get_control(self):
        return self.correction

class LaneNetFollower(Node):
    def __init__(self):
        super().__init__('lanenet_follower')
        self.bridge = CvBridge()

        # Initialize LaneNet model
        self.model = lanenet.LaneNet()
        self.model.load_weights('./data/models/lanenet_model')
        self.postprocessor = LaneNetPostProcessor()

        # Initialize variables
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.error = 0.0

        # Declare PID parameters
        self.declare_parameter('Kp', 0.5)  # Adjusted for metric distances
        self.declare_parameter('Kd', 0.1)
        self.declare_parameter('Ki', 0.001)
        self.declare_parameter('forward_speed', 1.0)

        Kp = self.get_parameter('Kp').value
        Kd = self.get_parameter('Kd').value
        Ki = self.get_parameter('Ki').value
        self.forward_speed = self.get_parameter('forward_speed').value

        self.pid_controller = PID(Kp, Kd, Ki)

        # Create subscription to camera images
        self.img_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10
        )

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create timer for publishing velocity commands (10ms = 100Hz)
        self.update_timer = self.create_timer(0.01, self.update_callback)

        # Camera parameters (adjust based on Zed2 calibration)
        self.focal_length = 700.0  # Approximate focal length in pixels
        self.camera_height = 0.5   # Camera height above ground in meters
        self.lane_width = 3.7      # Standard lane width in meters

    def pixel_to_metric(self, x, y, img_width, img_height):
        # Convert pixel coordinates to metric distances
        # Assume flat road, perspective projection
        z = self.camera_height * self.focal_length / (img_height - y)
        x_metric = z * (x - img_width / 2) / self.focal_length
        return x_metric, z

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_height, img_width = frame.shape[:2]

        # Preprocess image for LaneNet
        input_img = cv2.resize(frame, (512, 256))  # LaneNet input size
        input_img = input_img / 255.0

        # Run LaneNet inference
        binary_seg, instance_seg = self.model.predict(np.expand_dims(input_img, axis=0))

        # Post-process to get lane points
        lanes = self.postprocessor.post_process(binary_seg[0], instance_seg[0], input_img)

        # Select left and right lanes
        left_lane = None
        right_lane = None
        for lane in lanes:
            # Lane points are in original image coordinates (resized back)
            lane_points = [(int(x * img_width / 512), int(y * img_height / 256)) for x, y in lane['points']]
            if not lane_points:
                continue
            # Use bottom-most point for lane position
            x, y = lane_points[-1]
            x_metric, _ = self.pixel_to_metric(x, y, img_width, img_height)
            if x_metric < 0:  # Left lane
                if left_lane is None or y > left_lane[1]:
                    left_lane = (x, y, x_metric)
            else:  # Right lane
                if right_lane is None or y > right_lane[1]:
                    right_lane = (x, y, x_metric)

        # Fallback to previous lanes if not detected
        if left_lane is None:
            left_lane = self.prev_left_lane if self.prev_left_lane else (0, img_height, -self.lane_width / 2)
        if right_lane is None:
            right_lane = self.prev_right_lane if self.prev_right_lane else (img_width, img_height, self.lane_width / 2)

        self.prev_left_lane = left_lane
        self.prev_right_lane = right_lane

        # Compute midpoint in metric space
        mid_x_metric = (left_lane[2] + right_lane[2]) / 2

        # Compute cross-track error (desired position is lane center, i.e., mid_x_metric = 0)
        self.error = -mid_x_metric  # Negative if right of center, positive if left

        # Update PID controller
        self.pid_controller.update_control(self.error)

        # Visualization
        for lane in lanes:
            lane_points = [(int(x * img_width / 512), int(y * img_height / 256)) for x, y in lane['points']]
            for i in range(len(lane_points) - 1):
                cv2.line(frame, lane_points[i], lane_points[i + 1], (0, 255, 0), 2)
        if left_lane and right_lane:
            mid_x = int((left_lane[0] + right_lane[0]) / 2)
            mid_y = int((left_lane[1] + right_lane[1]) / 2)
            cv2.circle(frame, (mid_x, mid_y), 5, (0, 0, 255), -1)

        cv2.imshow('lanes', frame)
        cv2.waitKey(1)

    def update_callback(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = min(self.forward_speed, 1.0)
        cmd_vel.angular.z = self.pid_controller.get_control()
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = LaneNetFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
