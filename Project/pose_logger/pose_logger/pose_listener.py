import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import yaml
import os

class GoalPoseLogger(Node):
    def __init__(self):
        super().__init__('pose_logger')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.pose_log = []
        self.counter = 0

        # Register shutdown callback
        self.get_logger().info("GoalPoseLogger node started.")

    def listener_callback(self, msg):
        log_entry = {
            'id': self.counter,
            'action': 'navigate',
            'pose': {
                'position': {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w
                }
            }
        }
        self.counter += 1
        self.pose_log.append(log_entry)
        self.get_logger().info(f"Logged Pose #{self.counter}")

    def shutdown(self):
        filename = os.path.expanduser('~/goal_poses_log.yaml')
        with open(filename, 'w') as file:
            yaml.dump(self.pose_log, file)
        self.get_logger().info(f"Saved {len(self.pose_log)} poses to {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = GoalPoseLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
