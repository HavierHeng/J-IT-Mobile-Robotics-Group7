#!/usr/bin/env python

# Inspired by this approach: https://github.com/sinyeopgo/ROS-2-simulation/blob/main/my_pkg/src/lanefollowing.cpp
# Minor improvements include: 
# - PID controller instead of the hardcoded parameters - this allows us to tune it on the race track
# - Region of interest masking - to filter out any excess background blobs - we can do this by effective applying an AND bitmask via polyfill 0 - this is as we know that the race track blobs are only on the lower half of the image. We want to ensure we only ever detect 2 blobs for the most part.


# No lanenet/DL methods - The downside of this dumb method vs LaneNet DL model is that we cannot estimate the exact metric distance we are to the lanes in the world. Its an approximation via pixel distance, there is no scaling.

# Raw OpenCV - Take an image, greyscale and increase threshold to increase brightness of left and right lanes
# Abuses cv2.connectedComponentWithStats: https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
# In theory, there are only 2 main blobs/components at max - the left and right lane markers. This way, we can identify if we are to the left or right of the lane by taking their centroids. We can try to ensure the best candidates for left and right markers are tracked via a threshold
# CTE is calculated via taking the diff between the left and right centroids and the middle. 
# PID

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class PID:
    def __init__(self, Kp, Kd, Ki, dt=0.01):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.dt = dt  # delta time for PID calculations in seconds (e.g 0.01s is 10ms)
        self.prev_error = 0.0
        self.integral = 0.0
        self.correction = 0.0

    def update_control(self, current_error):
        # Calculate PID components
        self.integral += current_error * self.dt
        derivative = (current_error - self.prev_error) / self.dt
        self.correction = self.Kp * current_error + self.Kd * derivative + self.Ki * self.integral
        self.prev_error = current_error

    def get_control(self):
        return self.correction

class LaneFollowing(Node):
    def __init__(self):
        super().__init__('lanefollowing')
        # Initialize CvBridge - To convert ROS Image messages to CV messages for processing via OpenCV
        self.bridge = CvBridge()

        # Initialize variables
        self.prevpt1 = np.array([0.0, 0.0])
        self.prevpt2 = np.array([0.0, 0.0])
        self.error = 0.0

        # Declare PID parameters
        self.declare_parameter('Kp', 0.0225)  # Proportional gain 
        self.declare_parameter('Kd', 0.01)    # Derivative gain
        self.declare_parameter('Ki', 0.0001)  # Integral gain
        self.declare_parameter('forward_speed', 1)  # Forward speed - In theory, we would want throttle to be 1 (highest speed 1/10 car can tahan)

        Kp = self.get_parameter('Kp').value
        Kd = self.get_parameter('Kd').value
        Ki = self.get_parameter('Ki').value
        self.forward_speed = self.get_parameter('forward_speed').value

        self.pid_controller = PID(Kp, Kd, Ki)

        # Create subscription to camera images
        # Use the rectified image from the Zed2 camera
        self.img_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',  # Use the rectified color image from zed2 (accounts for camera parameters and distortion)
            self.image_callback,
            10
        )

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create timer for publishing velocity commands (10ms = 100Hz)
        self.update_timer = self.create_timer(0.01, self.update_callback)

    def create_roi_mask(self, height, width):
        # Create a trapezoidal mask for ROI
        mask = np.zeros((height, width), dtype=np.uint8)
        # Define trapezoid vertices
        top_width = int(width * 0.2)  # 20% of width at top
        bottom_width = int(width * 0.8)  # 80% of width at bottom
        top_y = int(height * 0.4)  # Start at 40% from top
        bottom_y = height  # Extend to bottom
        vertices = np.array([
            [(width - top_width) // 2, top_y],
            [(width + top_width) // 2, top_y],
            [(width + bottom_width) // 2, bottom_y],
            [(width - bottom_width) // 2, bottom_y]
        ], dtype=np.int32)
        # Fill the trapezoid with white (255)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Create and apply ROI mask
        mask = self.create_roi_mask(gray.shape[0], gray.shape[1])
        gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Adjust grayscale image
        gray = gray + 100 - np.mean(gray)
        _, gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        # Select ROI (lower third of the image, already masked)
        rows, cols = gray.shape
        dst = gray[rows // 3 * 2:rows, 0:cols]

        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=8)

        cpt = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        if num_labels > 1:
            # Track lane markers
            # ALgorithm works by picking the 2 candidates with the least possible distance from previous cached values for left and right markers (if they are within a reasonable threshold)
            mindistance1 = []
            mindistance2 = []
            for i in range(1, num_labels):
                p = centroids[i]
                ptdistance1 = abs(p[0] - self.prevpt1[0])
                ptdistance2 = abs(p[0] - self.prevpt2[0])
                mindistance1.append(ptdistance1)
                mindistance2.append(ptdistance2)

            threshdistance1 = min(mindistance1)
            threshdistance2 = min(mindistance2)

            minlb1 = np.argmin(mindistance1)
            minlb2 = np.argmin(mindistance2)

            cpt[0] = centroids[minlb1 + 1]
            cpt[1] = centroids[minlb2 + 1]

            # Use previous points if distance is too large
            if threshdistance1 > 100:
                cpt[0] = self.prevpt1
            if threshdistance2 > 100:
                cpt[1] = self.prevpt2]
        else:
            cpt[0] = self.prevpt1
            cpt[1] = self.prevpt2

        # Update previous points
        self.prevpt1 = cpt[0]
        self.prevpt2 = cpt[1]

        # Compute midpoint
        fpt = np.array([(cpt[0][0] + cpt[1][0]) / 2, (cpt[0][1] + cpt[1][1]) / 2 + rows // 3 * 2])

        # Convert ROI to color for visualization
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        # Draw circles for visualization
        cv2.circle(frame, (int(fpt[0]), int(fpt[1])), 2, (0, 0, 255), 2)
        cv2.circle(dst, (int(cpt[0][0]), int(cpt[0][1])), 2, (0, 0, 255), 2)
        cv2.circle(dst, (int(cpt[1][0]), int(cpt[1][1])), 2, (255, 0, 0), 2)

        # Compute cross-track error
        self.error = cols / 2 - fpt[0]

        # Update PID controller
        self.pid_controller.update_control(self.error)

        # Display images - for debugging
        cv2.imshow('camera', frame)
        cv2.imshow('gray', dst)
        cv2.waitKey(1)

    def update_callback(self):
        # Publish velocity commands
        cmd_vel = Twist()
        cmd_vel.linear.x = min(self.forward_speed, 1.0)  # Cap at full throttle of 1.0
        cmd_vel.angular.z = self.pid_controller.get_control()  # based on PID correction
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
