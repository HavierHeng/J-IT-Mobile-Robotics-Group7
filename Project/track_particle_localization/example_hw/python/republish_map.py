# Script to republish map topic at a consistent frequency with current time. Generated using ChatGPT
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy
from time import sleep
from rclpy.time import Time

class MapRepublisherNode(Node):
    def __init__(self, publish_frequency: float):
        super().__init__('map_republisher')
        
        # Create a subscriber to the '/map' topic
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Create a publisher to publish the same message
        self.publisher = self.create_publisher(
            OccupancyGrid,
            '/republished_map',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Store the message to be republished later
        self.latest_map = None

        # Set up a timer to periodically publish the latest map message
        self.timer = self.create_timer(1.0 / publish_frequency, self.publish_map)

    def map_callback(self, msg: OccupancyGrid):
        # Store the latest map message when it is received
        self.latest_map = msg
        self.get_logger().info('Received new map message.')

    def publish_map(self):
        if self.latest_map:
            self.latest_map.header.stamp = self.get_clock().now().to_msg()
            self.latest_map.header.frame_id = 'map'
            self.publisher.publish(self.latest_map)
            self.get_logger().info('Republishing map message.')

def main(args=None):
    rclpy.init(args=args)
    
    # Specify the frequency at which the map message should be republished (in Hz)
    publish_frequency = 1.0  # You can change this to whatever frequency you need

    # Create and spin the republisher node
    node = MapRepublisherNode(publish_frequency)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
