# import rclpy
from rclpy.node import Node
# from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
# from concurrent.futures import Future
# from rclpy.executors import MultiThreadedExecutor
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import String
# import matplotlib.pyplot as plt
# import numpy as np

class Publisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'online_eff_topic', 10)
        # self.timer = self.create_timer(1.0, self.publish_hello_world)  # 每1秒发布一次消息

        self.get_logger().info('Publisher node has been started.')

    # def publish_hello_world(self):
    #     msg = String()
    #     msg.data = 'Hello, World!'
    #     self.publisher_.publish(msg)
    #     self.get_logger().info('Published: "%s"' % msg.data)

