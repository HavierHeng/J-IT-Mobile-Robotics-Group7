import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2  # sudo apt install ros-$ROS_DISTRO-sensor-msgs-py

class CloudMapListener(Node):
    def __init__(self):
        super().__init__('cloud_map_listener')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rtabmap/cloud_map',
            self.callback,
            10
        )

    def callback(self, msg):
        xs, ys = [], []

        for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")):
            x, y, z = p
            xs.append(x)
            ys.append(y)

        if xs and ys:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            self.get_logger().info(f"Xmin: {xmin}, Xmax: {xmax}")
            self.get_logger().info(f"Ymin: {ymin}, Ymax: {ymax}")
        else:
            self.get_logger().warn("Received empty point cloud.")

def main(args=None):
    rclpy.init(args=args)
    node = CloudMapListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
