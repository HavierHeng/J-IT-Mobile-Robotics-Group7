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
        self.get_logger().info("LaneFollowing node started")
        # Initialize CvBridge - To convert ROS Image messages to CV messages for processing via OpenCV
        self.bridge = CvBridge()

        # Initialize variables
        self.error = 0.0

        # Declare PID parameters
        self.declare_parameter('Kp', 0.008)  # Proportional gain 
        self.declare_parameter('Kd', -0.000)    # Derivative gain
        self.declare_parameter('Ki', -0.000001)  # Integral gain
        self.declare_parameter('forward_speed', 1.0)  # Forward speed - In theory, we would want throttle to be 1 (highest speed 1/10 car can tahan)

        Kp = self.get_parameter('Kp').value
        Kd = self.get_parameter('Kd').value
        Ki = self.get_parameter('Ki').value
        self.forward_speed = self.get_parameter('forward_speed').value

        self.pid_controller = PID(Kp, Kd, Ki)

        # Cache old values
        self.left_lane = None
        self.right_lane = None

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

        # Create publisher for high contrast + ROI image + Hough lines
        self.masked_gray_img_pub = self.create_publisher(Image, 'debug/gray/masked_image', 5)

        # Create publisher for high contrast + ROI image + Hough lines
        self.masked_color_img_pub = self.create_publisher(Image, 'debug/color/masked_image', 5)

        # Create timer for publishing velocity commands (10ms = 100Hz)
        self.update_timer = self.create_timer(0.01, self.update_callback)

    def create_roi_mask(self, height, width):
        # Create a trapezoidal mask for ROI
        mask = np.zeros((height, width), dtype=np.uint8)
        # Define trapezoid vertices
        top_width = int(width * 1)  # 20% of width at top
        bottom_width = int(width * 1)  # 80% of width at bottom
        top_y = int(height * 0.5)  # Start at 40% from top
        bottom_y = int(height * 1)  # Extend to bottom
        vertices = np.array([
            [(width - top_width) // 2, top_y],
            [(width + top_width) // 2, top_y],
            [(width + bottom_width) // 2, bottom_y],
            [(width - bottom_width) // 2, bottom_y]
        ], dtype=np.int32)
        # Fill the trapezoid with white (255)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def average_line(self, lines):
        """
        Average lines - called on left or right group
        """
        if len(lines) == 0:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        return np.polyfit(y_coords, x_coords, 1)  # returns [slope, intercept]

    def image_callback(self, msg):
        """
        TODO:
        - Fix the ROI cropping
        - Tune it somehow (maybe using inRange, or some othe filter) for the actual race track
        - Find the center of the track
        - Actually make it move with the PIDs
        """
        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Create and apply ROI mask
        mask = self.create_roi_mask(gray_frame.shape[0], gray_frame.shape[1])
        gray_masked = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        color_masked = cv2.bitwise_and(frame, frame, mask=mask)
        # gray_masked = gray_frame
        # color_masked = frame

        # Gaussian blur to smooth image
        # Blurred HSV (Colour)
        blurred_color = cv2.GaussianBlur(color_masked, (5, 5), 0)  
        hsv = cv2.cvtColor(blurred_color, cv2.COLOR_BGR2HSV)
        # Blurred Gray (Grayscale)
        # blurred_gray = cv2.GaussianBlur(gray_masked, (5, 5), 0)
        blurred_gray = gray_masked

        # Use Canny edge detection - lower threshold, higher threshold
        # TODO: Tune and see which value works better
        hsv_edges = cv2.Canny(hsv, 75, 150)
        gray_edges = cv2.Canny(blurred_gray, 75, 150)

        # Detect lines using HoughLinesP (instead of HoughLine because of computational power)
        # Hough lines
        gray_lines = cv2.HoughLinesP(gray_edges, 1, np.pi / 180, threshold=50, maxLineGap=50)
        hsv_lines = cv2.HoughLinesP(hsv_edges, 1, np.pi / 180, threshold=50, maxLineGap=50)

        thicc = 2  # Line thiccness for drawing


        # Old implementation: This one just gets all hough lines so there is a lot of noise
        # # Draw lines on color image
        # if hsv_lines is not None:
        #     for line in hsv_lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(color_masked, (x1, y1), (x2, y2), (0, 255, 0), thicc) 

        # # Optional: draw lines on grayscale image
        # if gray_lines is not None:
        #     for line in gray_lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(gray_masked, (x1, y1), (x2, y2), 255, thicc)

        # New version: Filtered hough lines by gradient and pick them out by possible groups
        # Just need to use grayscale tbh
        left_lines = []
        right_lines = []

        # Filter & group lines by slope and position
        if gray_lines is not None:
            for line in gray_lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue  # skip vertical lines
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.1:
                    continue  # filter out near-horizontal lines
                if slope < 0 and x1 < frame.shape[1] // 2 and x2 < frame.shape[1] // 2:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0 and x1 > frame.shape[1] // 2 and x2 > frame.shape[1] // 2:
                    right_lines.append((x1, y1, x2, y2))

        left_fit = self.average_line(left_lines)
        right_fit = self.average_line(right_lines)

        height = frame.shape[0]
        y1 = int(height * 0.6)
        y2 = height

        if left_fit is not None:
            self.left_lane = left_fit  # Update cache
        if right_fit is not None:
            self.right_lane = right_fit  # update cache

        # if left_fit is not None and right_fit is not None:
        #     self.get_logger().info("Found left and right")

        # If both lines exist, compute vanishing point and CTE for PID
        # Vanishing point tells us the direction to chase
        if self.left_lane is not None and self.right_lane is not None:
            m1, b1 = self.left_lane
            m2, b2 = self.right_lane
            if m1 != m2:
                vp_y = int((b2 - b1) / (m1 - m2))
                vp_x = int(m1 * vp_y + b1)
                cv2.circle(color_masked, (vp_x, vp_y), 5, (0, 0, 255), -1)
                self.error = frame.shape[1] // 2 - vp_x
                self.pid_controller.update_control(self.error)


        # Draw avg left and right lines on color image (we calculate on gray, but color is easier to debug)
        # Avg left is green, avg right is blue
        if self.left_lane is not None:
            x1 = int(self.left_lane[0] * y1 + self.left_lane[1])
            x2 = int(self.left_lane[0] * y2 + self.left_lane[1])
            cv2.line(color_masked, (x1, y1), (x2, y2), (0, 255, 0), thicc)  # Green

        if self.right_lane is not None:
            x1 = int(self.right_lane [0] * y1 + self.right_lane [1])
            x2 = int(self.right_lane [0] * y2 + self.right_lane [1])
            cv2.line(color_masked, (x1, y1), (x2, y2), (255, 0, 0), thicc)  # Blue
        
        # Debug window if running on local machine
        # cv2.imshow("Gray Masked", gray_masked)
        # cv2.imshow("Color Masked with Lines", color_masked)
        # cv2.waitKey(1)


        # Publish masked grayscale image with lines
        # gray_msg = self.bridge.cv2_to_imgmsg(gray_masked, encoding='mono8')
        # self.masked_gray_img_pub.publish(gray_msg)

        # Publish masked color image with lines
        color_msg = self.bridge.cv2_to_imgmsg(color_masked, encoding='bgr8')
        self.masked_color_img_pub.publish(color_msg)

        # self.get_logger().info("Masked image published")

        # Combined image version
        # combined = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
        # ros_combined = self.bridge.cv2_to_imgmsg(combined, encoding='bgr8')
        # self.masked_img_pub.publish(ros_combined)


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
