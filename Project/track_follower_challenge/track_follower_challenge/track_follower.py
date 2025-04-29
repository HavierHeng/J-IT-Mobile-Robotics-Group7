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
from datetime import datetime

class PID:
    def __init__(self, Kp, Kd, Ki, dt=0.01):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.dt = dt  # delta time for PID calculations in seconds (e.g 0.01s is 10ms)
        self.prev_error = 0.0
        self.integral = 0.0
        self.correction = 0.0
        self.last_update = datetime.now()
        self.last_control = datetime.now()

    def update_control(self, current_error):
        # Calculate PID components
        self.integral += current_error * self.dt
        derivative = (current_error - self.prev_error) / self.dt
        self.correction = self.Kp * current_error + self.Kd * derivative + self.Ki * self.integral
        self.prev_error = current_error
        self.last_update = datetime.now()

    def get_control(self):
        self.last_control = datetime.now()
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
        self.declare_parameter('Kp', 0.015)  # Proportional gain 
        self.declare_parameter('Kd', 0.015)    # Derivative gain
        self.declare_parameter('Ki', 0.0000)  # Integral gain
        self.declare_parameter('forward_speed', 1.0)  # Forward speed - In theory, we would want throttle to be 1 (highest speed 1/10 car can tahan)

        Kp = self.get_parameter('Kp').value
        Kd = self.get_parameter('Kd').value
        Ki = self.get_parameter('Ki').value
        self.bias = 9.0  # our car has funnjy drift not sure if linear - higher means it turns more left- 9.0 for straights
        self.forward_speed = self.get_parameter('forward_speed').value
        # 8.5 straights, 9.0 for turns
        self.pid_controller = PID(Kp, Kd, Ki)

        # Cache old values
        self.left_lane = None
        self.right_lane = None

        # Threshold for when to update error
        self.max_error_change = 20  # How much max pixel jump between updates

        # Create subscription to camera images
        # Use the rectified image from the Zed2 camera
        self.img_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',  # Use the rectified color image from zed2 (accounts for camera parameters and distortion)
            self.image_callback,
            10
        )

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 1)

        # Create publisher for high contrast + ROI image + Hough lines
        self.masked_gray_img_pub = self.create_publisher(Image, 'debug/gray/masked_image', 5)

        # Create publisher for high contrast + ROI image + Hough lines
        self.masked_color_img_pub = self.create_publisher(Image, 'debug/color/masked_image', 5)

        # Create timer for publishing velocity commands (10ms = 100Hz)
        self.update_timer = self.create_timer(0.01, self.update_callback)

    def create_roi_mask(self, height, width):
        # Create a trapezoidal mask + rectangle for ROI
        # This funny shape means that at the bottom the car can see most things (since we're certain that has a lot of lane)
        # the top trapezoid prevents false positive from items at vanishing points
        mask = np.zeros((height, width), dtype=np.uint8)

        #  RECTANGLE BOTTOM (e.g. bottom 20% of image)
        rect_top_y = int(height * 0.7)
        cv2.rectangle(mask, (0, rect_top_y), (width, height), 255, -1)

        #  TRAPEZOID ABOVE RECTANGLE
        top_width = int(width * 0.2)   # Narrow top
        bottom_width = width           # Full width at the base of trapezoid
        top_y = int(height * 0.57)      # Start trapezoid higher
        bottom_y = rect_top_y          # Connect to top of rectangle

        trapezoid = np.array([
            [(width - top_width) // 2, top_y],
            [(width + top_width) // 2, top_y],
            [(width + bottom_width) // 2, bottom_y],
            [(width - bottom_width) // 2, bottom_y]
        ], dtype=np.int32)

        cv2.fillPoly(mask, [trapezoid], 255)

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
        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Test: Resize frame to drop latency
        # frame = cv2.resize(frame, (320, 240))
        # gray_frame = cv2.resize(gray_frame, (320, 240))

        # Create and apply ROI mask
        mask = self.create_roi_mask(gray_frame.shape[0], gray_frame.shape[1])
        gray_masked = gray_frame  # I laze change name haha
        # color_masked = cv2.bitwise_and(frame, frame, mask=mask)
        # gray_masked = gray_frame
        # color_masked = frame

        # Gaussian blur to smooth image
        # Blurred HSV (Colour)
        # blurred_color = cv2.GaussianBlur(color_masked, (5, 5), 0)  
        # hsv = cv2.cvtColor(blurred_color, cv2.COLOR_BGR2HSV)
        # Blurred Gray (Grayscale)
        # gray_masked = cv2.GaussianBlur(gray_masked, (5, 5), 1)

        # Note to future: bilateral 5/6 is okay for the normal part of the track
        # Don't ask why. Do not comment out the bilateral filter else noisy.
        gray_masked = cv2.bilateralFilter(gray_masked, 6, 25, 25)  # makes colour similoar or something - easier to contrast and group similar greys

        cv2.imshow("bilateral filter", gray_masked)
        cv2.waitKey(1)

        # Enhance contrast: Contrast stretching
        min_val, max_val = np.percentile(gray_masked[mask > 0], [50, 98])  # Avoid outliers - higher the first value, the more aggressive it changes greys to blacks (black point), higher second value means that less whitish pixels are whites
        if max_val > min_val:  # Avoid division by zero
            gray_contrast = np.clip((gray_masked - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
        else:
            gray_contrast = gray_masked

        # Optional: Apply CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_contrast = clahe.apply(gray_contrast)


        # Use Canny edge detection - lower threshold, higher threshold
        # hsv_edges = cv2.Canny(hsv, 75, 150)
        gray_edges_full = cv2.Canny(gray_contrast, 100, 150)
        # Apply mask after so canny cannot pick ROI edges as candidate left/right lines
        gray_edges = cv2.bitwise_and(gray_edges_full, gray_edges_full, mask=mask)

        # Detect lines using HoughLinesP (instead of HoughLine because of computational power)
        # Hough lines - threshold is for removing weak lines, maxLineGap is how many small lines to combine
        # threshold 65 updates fast enough without too much shaking of the vanishing point
        # max linbe gap 30 gives us enough hough lines groupss to average
        gray_lines = cv2.HoughLinesP(gray_edges, 1, np.pi / 180, threshold=65, maxLineGap=30)
        # hsv_lines = cv2.HoughLinesP(hsv_edges, 1, np.pi / 180, threshold=50, maxLineGap=50)

        thicc = 2  # Line thiccness for drawing

        # Old implementation: This one just gets all hough lines so there is a lot of noise
        # # Draw lines on color image
        # if hsv_lines is not None:
        #     for line in hsv_lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(color_masked, (x1, y1), (x2, y2), (0, 255, 0), thicc) 
        masked_hough = cv2.bitwise_and(frame, frame, mask=mask)  # debug what mask see
        # Optional: draw lines on grayscale image
        if gray_lines is not None:
            for line in gray_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(masked_hough, (x1, y1), (x2, y2), (255, 128, 128), thicc)
        cv2.imshow("hough with mask", masked_hough)
        # cv2.imshow("hough with mask", gray_contrast) 
        cv2.waitKey(1)


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
                if abs(slope) < 0.15:
                    continue  # filter out near-horizontal lines

                # Slacken the condition to be having 3/4 on either side, and correct gradient
                if slope < 0 and x1 < frame.shape[1] * 3 // 4 and x2 < frame.shape[1] * 3// 4:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0 and x1 > frame.shape[1] // 4 and x2 > frame.shape[1] // 4:
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
        if self.left_lane is None: 
            #TODO: add logic to pull back based on None and send cmd -> return
            self.error = -20 + self.bias
        if self.right_lane is None:
            self.error = 15 + self.bias

        if self.left_lane is None and self.right_lane is None:
            self.error = self.bias


        # If both lines exist, compute vanishing point and CTE for PID
        # Vanishing point tells us the direction to chase
        if self.left_lane is not None and self.right_lane is not None:
            m1, b1 = self.left_lane
            m2, b2 = self.right_lane
            if m1 != m2:
                vp_y = int((b2 - b1) / (m1 - m2))
                vp_x = int(m1 * vp_y + b1)
                cv2.circle(frame, (vp_x, vp_y), 5, (0, 0, 255), -1)
                new_error = gray_frame.shape[1] // 2 - vp_x
                # Limit error change via thresholding how much a vanishing point can jump
                # if abs(new_error - self.error) > self.max_error_change:
                #     new_error = self.error + np.sign(new_error - self.error) * self.max_error_change
                self.error = new_error + self.bias
                # self.pid_controller.update_control(self.error)


        # Draw avg left and right lines on color image (we calculate on gray, but color is easier to debug)
        # Avg left is green, avg right is blue
        if self.left_lane is not None:
            x1 = int(self.left_lane[0] * y1 + self.left_lane[1])
            x2 = int(self.left_lane[0] * y2 + self.left_lane[1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thicc)  # Green

        if self.right_lane is not None:
            x1 = int(self.right_lane [0] * y1 + self.right_lane [1])
            x2 = int(self.right_lane [0] * y2 + self.right_lane [1])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thicc)  # Blue


        # Publish masked grayscale image with lines
        # gray_msg = self.bridge.cv2_to_imgmsg(gray_masked, encoding='mono8')
        # self.masked_gray_img_pub.publish(gray_msg)

        # Publish masked color image with lines
        color_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.masked_color_img_pub.publish(color_msg)

        # Debug Contrast + ROI

        cv2.imshow("hough lines", frame)
        cv2.waitKey(1)


        # Combined image version
        # combined = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
        # ros_combined = self.bridge.cv2_to_imgmsg(combined, encoding='bgr8')
        # self.masked_img_pub.publish(ros_combined)


    def update_callback(self):
        # Publish velocity commands every control cycle
        cmd_vel = Twist()
        cmd_vel.linear.x = min(self.forward_speed, 1.0)  # Cap at full throttle of 1.0
        self.pid_controller.update_control(self.error)  # Reasoning for this is that the control loop is a lot faster than the image loop - some of the PID calculations e.g integral and derivation evolve over time
        correction = self.pid_controller.get_control()  # based on PID correction
        # self.get_logger().info(f"error: {self.pid_controller.prev_error} angular_z: {correction} latency: {(self.pid_controller.last_control - self.pid_controller.last_update).total_seconds() * 1000}")
        cmd_vel.angular.z = max(min(correction, 3.0), -3.0)  # clamp just in case, roughly Pi radians
        self.get_logger().info(f"error: {self.pid_controller.prev_error} angular_z: {cmd_vel.angular.z} rad/s")      
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
