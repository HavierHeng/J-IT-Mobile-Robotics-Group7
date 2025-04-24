#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from lane_detector.msg import Lane_Image  # Custom image with mask

# LaneNet implementation only works with TF 1.x
tf_compat = tf.compat.v1
tf_compat.disable_v2_behavior()

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

        super().__init__('lanenet_follower')

        # Declare parameters
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')  # For updating the camera parameters later
        self.declare_parameter('output_image', '/lanenet/image')
        self.declare_parameter('output_lane', '/lanenet/lane_image')  # For images with lane mask
        self.declare_parameter('weight_path', './data/models/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('Kp', 0.5)
        self.declare_parameter('Kd', 0.1)
        self.declare_parameter('Ki', 0.001)
        self.declare_parameter('forward_speed', 1.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_image = self.get_parameter('output_image').value
        self.output_lane = self.get_parameter('output_lane').value
        self.weight_path = self.get_parameter('weight_path').value
        self.use_gpu = self.get_parameter('use_gpu').value
        Kp = self.get_parameter('Kp').value
        Kd = self.get_parameter('Kd').value
        Ki = self.get_parameter('Ki').value
        self.forward_speed = self.get_parameter('forward_speed').value

        # Validate weight path
        if not os.path.exists(self.weight_path):
            self.get_logger().error(f"Model weights not found at {self.weight_path}")
            raise FileNotFoundError(f"Model weights not found at {self.weight_path}")

        self.bridge = CvBridge()

        # Initialize LaneNet model
        self.init_lanenet()

        # Initialize variables
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.error = 0.0

        # Camera parameters - these will be updated later when CameraInfo is received from Zed2
        self.focal_length = 700.0  # Default until CameraInfo is received
        self.principal_point_x = 0.0
        self.camera_info_received = False

        self.pid_controller = PID(Kp, Kd, Ki)

        # Create subscription to camera images
        self.img_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        # Create subscription to camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )


        # Create publisher for LaneNet output image
        self.pub_image = self.create_publisher(Image, self.output_image, 10)
        self.pub_laneimage = self.create_publisher(Lane_Image, self.lane_image_topic, 1)

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create timer for publishing velocity commands (10ms = 100Hz)
        self.update_timer = self.create_timer(0.01, self.update_callback)

    def camera_info_callback(self, msg):
        # Update camera parameters from CameraInfo
        # Calibration matrix format (row order based on ROS2 docs): 
        # [fx, 0, cx, 0, fy, cy, 0, 0, 1]

        self.focal_length_x = msg.k[0]  # fx
        self.focal_length_y = msg.k[4]  # fy

        self.principal_point_x = msg.k[2]  # cx
        self.principal_point_y = msg.k[5]  # cy
        self.camera_info_received = True

    def init_lanenet(self):
        '''
        Initialize the tensorflow model. Yes I ripped this off from the original Node implementation.
        '''
        self.input_tensor = tf_compat.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf_compat.constant('test', tf.string)
        net = lanenet.LaneNet(phase='test', net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()

        saver = tf_compat.train.Saver()
        sess_config = tf_compat.ConfigProto()
        if self.use_gpu:
            sess_config.gpu_options.allow_growth = True
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf_compat.Session(config=sess_config)
        saver.restore(sess=self.sess, save_path=self.weight_path)

    def pixel_to_metric(self, x, y, img_width, img_height):
        """
        Convert pixel coordinates to metric distances.
        Metric x defines the position to the right of the camera's optical axis. Useful for telling of a point is left or right of the camera axis (e.g lanes).
        z is the depth of the point from the camera.
        Assumes flat road, perspective projection based estimation.
        """
        if not self.camera_info_received:
            self.get_logger().warn("CameraInfo not received, using default focal length which may be wrong...")

        # To get z depth, from camera to pixel in real world
        # Can use the focal length to solve using either x or y principle point and focal length
        # Can assume that camera is isotrophic - focal length on x and y is the same
        z = self.camera_height * self.focal_length_y / (img_height - y)

        # To get the real world X, this formula accounts for pers proj
        # scale by z depth, and adjust the position by the principal point x
        x_metric = z * (x - self.principal_point_x) / self.focal_length
        return x_metric, z

    def preprocess(self, img):
        """
        Images have to be sized to (512, 256) for input into model
        """
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        return image

    def inference_net(self, img, original_img):
        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_ret, self.instance_seg_ret],
            feed_dict={self.input_tensor: [img]}
        )

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=original_img
        )

        mask_image = postprocess_result['mask_image']
        mask_image = cv2.resize(mask_image, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.addWeighted(original_img, 0.6, mask_image, 0.4, 0)
        return mask_image, postprocess_result['lane_points']

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_height, img_width = frame.shape[:2]
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return  # Skip processing of image for this round

        original_img = frame.copy()
        resized_image = self.preprocess(frame)
        mask_image, lane_points = self.inference_net(resized_image, original_img)

        try:
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, encoding="bgr8")
            self.pub_image.publish(out_img_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error publishing image: {e}")

        # Process lane points
        # If all goes well, these should be points representing the candidate points for left and right lane. 
        # Choose the left or right lane based on their position in space
        left_lane = None
        right_lane = None
        for lane in lane_points:
            lane_coords = [(int(x * img_width / 512), int(y * img_height / 256)) for x, y in lane]
            if not lane_coords:
                continue
            x, y = lane_coords[-1]  # Bottom-most point
            x_metric, _ = self.pixel_to_metric(x, y, img_width, img_height)
            if x_metric < 0:  # Left lane
                if left_lane is None or y > left_lane[1]:
                    left_lane = (x, y, x_metric)
            else:  # Right lane
                if right_lane is None or y > right_lane[1]:
                    right_lane = (x, y, x_metric)

        # Fallback to previous lanes if unable to grab a left or right lane candidate
        if left_lane is None:
            left_lane = self.prev_left_lane if self.prev_left_lane else (0, img_height, -self.lane_width / 2)
        if right_lane is None:
            right_lane = self.prev_right_lane if self.prev_right_lane else (img_width, img_height, self.lane_width / 2)

        self.prev_left_lane = left_lane
        self.prev_right_lane = right_lane

        # Compute midpoint in metric space
        mid_x_metric = (left_lane[2] + right_lane[2]) / 2

        # Compute cross-track error
        self.error = -mid_x_metric

        # Update PID controller to recalculate angular z turning
        self.pid_controller.update_control(self.error)

        # Visualization in OpenCV2 window - just for debugging
        for lane in lane_points:
            lane_coords = [(int(x * img_width / 512), int(y * img_height / 256)) for x, y in lane]
            for i in range(len(lane_coords) - 1):
                cv2.line(frame, lane_coords[i], lane_coords[i + 1], (0, 255, 0), 2)
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

    def destroy_node(self):
        self.sess.close()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LaneNetFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
