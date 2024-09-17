import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from concurrent.futures import Future
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from sensor_msgs.msg import String
import matplotlib.pyplot as plt
import numpy as np

class HelloWorldPublisher(Node):
    def __init__(self):
        super().__init__('hello_world_publisher')
        self.publisher_ = self.create_publisher(String, 'hello_world_topic', 10)
        self.timer = self.create_timer(1.0, self.publish_hello_world)  # 每1秒发布一次消息
        self.get_logger().info('HelloWorldPublisher node has been started.')

    def publish_hello_world(self):
        msg = String()
        msg.data = 'Hello, World!'
        self.publisher_.publish(msg)
        self.get_logger().info('Published: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = HelloWorldPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()