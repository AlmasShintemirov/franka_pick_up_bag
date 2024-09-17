import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from concurrent.futures import Future
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.subscription = None
        self.data = None
        self.future = None

    def subscribe_once(self, topic_name='/Camera_rgb'):
        if self.subscription is None:
            if topic_name == '/language_topic':
                self.subscription = self.create_subscription(
                    String,
                    topic_name,
                    self.callback,
                    10
                )
            else:
                self.subscription = self.create_subscription(
                    Image,
                    topic_name,
                    self.callback,
                    10
                )
            self.future = Future()

        return self.future

    def callback(self, msg):
        self.data = msg
        self.get_logger().info('Received data')
        
        self.future.set_result(msg)
        
        self.destroy_subscription(self.subscription)
        self.subscription = None

def sub_call(node, topic='/Camera_rgb'):
    future = node.subscribe_once(topic)

    executor = MultiThreadedExecutor()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        
        if future.done():
            break
    return future.result()

# class LanguageSubscriber(Node):

#     def __init__(self):
#         super().__init__('language_subscriber')
        
#         self.subscription = self.create_subscription(
#             String,
#             'language_topic',
#             self.listener_callback,
#             10)
#         self.subscription

#     def listener_callback(self, msg):
#         self.get_logger().info('Received: "%s"' % msg.data)
